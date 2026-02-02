"""
Script pour generer le PDF du rapport final Rakuten
avec le meme rendu que l'HTML en utilisant Playwright.
"""

import subprocess
import sys
import os
import asyncio

# Install playwright if needed
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.async_api import async_playwright

async def generate_pdf():
    # Paths
    base_dir = r"D:\datascientest\workspace\OCT25_BMLE_RAKUTEN_WS\repo\OCT25_BMLE_RAKUTEN"
    html_path = os.path.join(base_dir, "reports", "RAPPORT_FINAL_RAKUTEN.html")
    pdf_path = os.path.join(base_dir, "reports", "RAPPORT_FINAL_RAKUTEN.pdf")

    # Convert to file URL
    file_url = f"file:///{html_path.replace(os.sep, '/')}"

    print(f"Opening: {file_url}")

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Navigate to HTML file
        await page.goto(file_url, wait_until="networkidle")

        # Wait for fonts to load
        await page.wait_for_timeout(2000)

        # Generate PDF with proper margins
        await page.pdf(
            path=pdf_path,
            format="A4",
            margin={
                "top": "20mm",
                "right": "18mm",
                "bottom": "20mm",
                "left": "18mm"
            },
            print_background=True,
            prefer_css_page_size=True
        )

        await browser.close()

    print(f"[OK] PDF genere avec succes: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    asyncio.run(generate_pdf())
