# scrape_perilous_ok.py
import os, random, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

OUT_DIR = "data/perilous"
os.makedirs(OUT_DIR, exist_ok=True)

opts = Options()
opts.add_argument("--headless=new")       # keep headless
# DO NOT add --disable-gpu  ⇦  let Chrome keep its GPU
opts.add_argument("--window-size=1600,1200")
driver = webdriver.Chrome(options=opts)

N = 200
for _ in range(N):
    seed = random.randint(0, 999_999)
    url  = f"https://watabou.github.io/perilous-shores/?seed={seed}"
    driver.get(url)

    time.sleep(5)                         # plenty of time to draw

    # hide UI bits so the map is clean
    driver.execute_script("""
        document.querySelectorAll('.controls, .menu, .legend')
                .forEach(e => e.style.display = 'none');
    """)

    # full‑page screenshot (map fills viewport)
    fp = os.path.join(OUT_DIR, f"perilous_{seed}.png")
    driver.save_screenshot(fp)
    print("saved", fp)

driver.quit()