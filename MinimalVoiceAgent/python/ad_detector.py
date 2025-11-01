from datetime import datetime, timedelta
import sys
import argparse
import os
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import json
import gc
import re
import mss  # For screen capture
import pyautogui  # For mouse actions
import time
import glob

# ANSI escape codes
GREEN = "\033[92m"
RED = "\033[91m"
GOLD = "\033[93m"
RESET = "\033[0m"

# Constants
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
SKIP_AD_PATTERNS = [r"(?i)skip"]  # Updated: Match "Skip" only, case insensitive
MAX_BBOX_AREA = 50000  # Max area in pixels for valid button (~200x250px max)
MIN_CONFIDENCE = 0.5  # Min score for detection (if model outputs scores)

def load_settings(settings_file='settings.json'):
    try:
        if not os.path.exists(settings_file):
            print(f"Error: {settings_file} does not exist.")
            return None
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            if not settings.get('grounding_caption'):
                print("Error: 'grounding_caption' not found in settings.json (e.g., {'grounding_caption': 'Skip'})")
                return None
            return settings
    except Exception as e:
        print(f"Error loading settings: {e}")
        return None

def capture_screen():
    """Capture full screen with MSS and return PIL Image and size (native res)."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        width, height = screenshot.size
        return img, (width, height)

def process_image(image_path):
    """Load an image from file and return PIL Image and size (native res)."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    return img, (width, height)

def detect_skip_button(model, processor, device, dtype, grounding_caption, image, image_size):
    """Detect 'Skip' button using Florence-2 CAPTION_TO_PHRASE_GROUNDING (full res)."""
    try:
        prompt = GROUNDING_TASK + grounding_caption  # e.g., "<CAPTION_TO_PHRASE_GROUNDING> Skip"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_result = processor.post_process_generation(generated_text, task=GROUNDING_TASK, image_size=image_size)
        
        grounding_result = parsed_result.get(GROUNDING_TASK, {})
        bboxes = grounding_result.get('bboxes', [])
        labels = grounding_result.get('labels', [])
        
        detected_bbox = None
        for box, label in zip(bboxes, labels):
            label_stripped = label.strip()
            for pattern in SKIP_AD_PATTERNS:
                if re.search(pattern, label_stripped):
                    # Check bbox area for validity
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > MAX_BBOX_AREA:
                        print(f"{RED}Ignoring large bbox {box} with area {area} > {MAX_BBOX_AREA}{RESET}")
                        continue
                    # Optional: Confidence if available (Florence outputs scores in some modes)
                    # Assume detected if matched
                    detected_bbox = box
                    x_center = int((box[0] + box[2]) / 2)
                    y_center = int((box[1] + box[3]) / 2)
                    print(f"{GOLD}Detected '{label_stripped}' at center ({x_center}, {y_center}) with bbox {box}{RESET}")
                    return (x_center, y_center), detected_bbox
        
        print(f"{RED}No valid 'Skip' button detected.{RESET}")
        return None, None
    except Exception as e:
        print(f"{RED}Error detecting skip button: {e}{RESET}")
        return None, None

def draw_bbox_on_image(image, bbox, output_path):
    """Draw the bounding box on the image and save it for verification."""
    if bbox:
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        image.save(output_path)
        print(f"{GOLD}Saved image with bbox to {output_path}{RESET}")

def skip_ad_action(position):
    """Simulate clicking at the given position (cross-platform via pyautogui)."""
    if position:
        x, y = position
        pyautogui.click(x, y)
        print(f"{GREEN}Clicked at ({x}, {y}) to skip ad.{RESET}")

def verify_skip_success(model, processor, device, dtype, grounding_caption):
    """Post-click verification: Re-capture and check if button is gone."""
    time.sleep(2)  # Brief wait for UI update
    image, image_size = capture_screen()
    position, bbox = detect_skip_button(model, processor, device, dtype, grounding_caption, image, image_size)
    success = bbox is None
    status = "success" if success else "partial"
    message = "Ad skipped successfully!" if success else "Clicked, but 'Skip' button may still be present—manual check needed."
    print(f"{GREEN if success else RED}{message}{RESET}")
    return status, message

def run_test_mode(model, processor, device, dtype, grounding_caption, test_dir, output_dir):
    """Process images in test_dir, detect button, draw bbox if found, save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(test_dir, "*.png")) + glob.glob(os.path.join(test_dir, "*.jpg"))
    
    results = []
    for img_path in image_files:
        print(f"Processing {img_path}")
        image, image_size = process_image(img_path)
        position, bbox = detect_skip_button(model, processor, device, dtype, grounding_caption, image, image_size)
        result = {"file": img_path, "detected": bbox is not None, "position": position, "bbox": bbox}
        results.append(result)
        if bbox:
            output_path = os.path.join(output_dir, f"output_{os.path.basename(img_path)}")
            draw_bbox_on_image(image.copy(), bbox, output_path)
        else:
            print(f"{RED}No detection for {img_path}{RESET}")
    
    print(f"{GOLD}Test complete: {sum(1 for r in results if r['detected'])}/{len(results)} detections.{RESET}")
    return results

def run_live_mode(model, processor, device, dtype, grounding_caption, interval=3, max_poll_seconds=30, output_dir='live_outputs'):
    """Live polling loop: Detect, click if found, verify."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    button_found = False
    click_position = None
    
    print(f"{GOLD}Starting live ad skip monitoring. Interval: {interval}s, Max poll: {max_poll_seconds}s.{RESET}")
    
    while (time.time() - start_time) < max_poll_seconds:
        image, image_size = capture_screen()
        position, bbox = detect_skip_button(model, processor, device, dtype, grounding_caption, image, image_size)
        
        if bbox:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"detection_{timestamp}.png")
            draw_bbox_on_image(image.copy(), bbox, output_path)
            
            click_position = position
            button_found = True
            break  # Click once found
        
        print(f"{RED}No button in this poll.{RESET}")
        time.sleep(interval)
    
    if not button_found:
        print(f"{RED}No 'Skip' button found within {max_poll_seconds}s.{RESET}")
        return {"status": "partial", "message": "No skip ad button detected within time limit. Try again?"}
    
    # Click
    skip_ad_action(click_position)
    
    # Verify
    status, message = verify_skip_success(model, processor, device, dtype, grounding_caption)
    
    return {"status": status, "message": message, "clicked_at": click_position}

def main():
    parser = argparse.ArgumentParser(description="Florence-2 based YouTube ad skipper with test/live modes.")
    parser.add_argument("--mode", type=str, choices=["test", "live"], default="live", help="Mode: test (process images) or live (poll/click).")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory of test images (for --mode test).")
    parser.add_argument("--output-dir", type=str, default="test_outputs", help="Output dir for bboxes/saves.")
    parser.add_argument("--interval", type=int, default=3, help="Poll interval in seconds (live mode).")
    parser.add_argument("--max-poll-seconds", type=int, default=30, help="Max poll duration in seconds (live mode).")
    args = parser.parse_args()
    
    settings = load_settings()
    if not settings:
        sys.exit(1)
    
    grounding_caption = settings['grounding_caption']  # e.g., "Skip"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    try:
        model = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
        
        if args.mode == "test":
            if not args.test_dir:
                print("Error: --test-dir required for test mode.")
                sys.exit(1)
            results = run_test_mode(model, processor, device, dtype, grounding_caption, args.test_dir, args.output_dir)
            print(json.dumps(results, indent=2))  # JSON output for scripting
        else:  # live
            result = run_live_mode(model, processor, device, dtype, grounding_caption, args.interval, args.max_poll_seconds, args.output_dir)
            print(json.dumps(result, ensure_ascii=False))  # JSON status for C#
    
    except Exception as e:
        print(f"{RED}Fatal error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup
        del model
        del processor
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        print(f"{RESET}Script ended.")

if __name__ == "__main__":
    main()