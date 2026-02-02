"""
Script pour capturer des screenshots de l'application Streamlit
Usage: python scripts/capture_screenshots.py
"""

import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "playwright", "-q"])
    subprocess.run(["playwright", "install", "chromium"])
    from playwright.async_api import async_playwright

# Configuration
BASE_URL = "http://localhost:8501"
OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pages to capture
PAGES = [
    {"name": "home", "url": "/", "wait": 3000},
    {"name": "donnees", "url": "/Données", "wait": 3000},
    {"name": "preprocessing", "url": "/Preprocessing", "wait": 3000},
    {"name": "modeles", "url": "/Modèles", "wait": 3000},
    {"name": "demo", "url": "/Démo", "wait": 3000},
    {"name": "performance", "url": "/Performance", "wait": 3000},
    {"name": "conclusions", "url": "/Conclusions", "wait": 3000},
]


async def capture_screenshots():
    """Capture screenshots of all Streamlit pages."""
    print(f"Output directory: {OUTPUT_DIR}")

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1.5
        )
        page = await context.new_page()

        for page_info in PAGES:
            try:
                url = f"{BASE_URL}{page_info['url']}"
                print(f"Capturing {page_info['name']}... ({url})")

                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(page_info['wait'])

                # Hide Streamlit menu for cleaner screenshots
                await page.evaluate("""
                    const menu = document.querySelector('[data-testid="stMainMenu"]');
                    if (menu) menu.style.display = 'none';
                    const header = document.querySelector('[data-testid="stHeader"]');
                    if (header) header.style.display = 'none';
                """)

                output_path = OUTPUT_DIR / f"streamlit_{page_info['name']}.png"
                await page.screenshot(path=str(output_path), full_page=True)
                print(f"  Saved: {output_path}")

            except Exception as e:
                print(f"  Error capturing {page_info['name']}: {e}")

        await browser.close()

    print(f"\nScreenshots saved to: {OUTPUT_DIR}")


async def check_streamlit_running():
    """Check if Streamlit is running."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(BASE_URL, timeout=5) as response:
                return response.status == 200
    except:
        return False


async def main():
    # Check if Streamlit is running
    print("Checking if Streamlit is running...")

    if not await check_streamlit_running():
        print(f"\nStreamlit is NOT running at {BASE_URL}")
        print("Please start it first with:")
        print("  cd src/streamlit && streamlit run app.py")
        print("\nOr run in another terminal:")
        print("  streamlit run src/streamlit/app.py")
        return

    print("Streamlit is running. Starting capture...")
    await capture_screenshots()


if __name__ == "__main__":
    # Install aiohttp if needed
    try:
        import aiohttp
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])

    asyncio.run(main())
