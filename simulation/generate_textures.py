"""Generate high-quality realistic procedural textures for MuJoCo simulation."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEXTURE_DIR = Path(__file__).resolve().parent / "models" / "textures"
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)


def generate_wood_texture(width=1024, height=1024) -> Path:
    """Generate a realistic warm oak wood tabletop texture with grain and subtle rings."""
    # Base warm wood color tones (RGB)
    base_color = np.array([215, 170, 125], dtype=np.float32)
    dark_grain = np.array([160, 110, 70], dtype=np.float32)
    
    # Coordinate grid
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    
    # Wood grain mathematical synthesis: longitudinal grain + slight wobble + rings
    freq = 0.035
    wobble = np.sin(y * 0.015) * 8.0 + np.sin(y * 0.05) * 3.0
    r = np.abs(x - width * 0.45 + wobble)
    ring_pattern = np.sin(r * freq)
    
    # High-frequency fiber noise
    rng = np.random.default_rng(42)
    fine_noise = rng.normal(0, 0.08, (height, width))
    fine_noise = np.clip(fine_noise, -0.2, 0.2)
    
    # Combine patterns
    t = 0.5 * (ring_pattern + 1.0)
    t = np.power(t, 1.4)  # sharpen grain lines
    t = np.clip(t + fine_noise, 0.0, 1.0)
    
    img_data = (1.0 - t[:, :, None]) * base_color + t[:, :, None] * dark_grain
    img_data = np.clip(img_data, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_data)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    
    out_path = TEXTURE_DIR / "wood_table.png"
    img.save(str(out_path), "PNG")
    print(f"Generated wood texture: {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def generate_lab_floor_texture(width=1024, height=1024) -> Path:
    """Generate a clean, modern slate-grey laboratory floor with subtle grid lines."""
    base_rgb = np.full((height, width, 3), 42, dtype=np.uint8)  # sleek dark slate #2a2a2a
    
    img = Image.fromarray(base_rgb)
    draw = ImageDraw.Draw(img)
    
    # Draw subtle grid lines
    grid_spacing = 128
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=(58, 62, 68), width=2)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=(58, 62, 68), width=2)
        
    out_path = TEXTURE_DIR / "lab_floor.png"
    img.save(str(out_path), "PNG")
    print(f"Generated floor texture: {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def generate_workbench_mat_texture(width=1024, height=1024) -> Path:
    """Generate an ESD antistatic workbench mat texture (classic matte blue/grey with border)."""
    # Classic antistatic workbench mat color (muted deep slate-blue #2d3748 or green/blue)
    mat_rgb = np.full((height, width, 3), [45, 55, 72], dtype=np.uint8)
    
    # Subtle surface stippling
    rng = np.random.default_rng(123)
    noise = rng.integers(-4, 5, size=(height, width, 3), dtype=np.int16)
    mat_data = np.clip(mat_rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(mat_data)
    draw = ImageDraw.Draw(img)
    
    # Subtle border rim
    border = 16
    draw.rectangle([border, border, width - border, height - border], outline=(65, 80, 105), width=3)
    
    out_path = TEXTURE_DIR / "workbench_mat.png"
    img.save(str(out_path), "PNG")
    print(f"Generated workbench mat texture: {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


if __name__ == "__main__":
    generate_wood_texture()
    generate_lab_floor_texture()
    generate_workbench_mat_texture()
