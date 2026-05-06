import { chromium } from "playwright";

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [k, ...rest] = arg.replace(/^--/, "").split("=");
    return [k, rest.join("=") || "true"];
  }),
);

const lat = args.lat || "1.2966";
const lon = args.lon || "103.7764";
const out = args.out || "data/rendered/streets_gl.png";
const url = args.url || `https://streets.gl/#map=17/${lat}/${lon}`;

const browser = await chromium.launch({
  headless: true,
  args: ["--ignore-gpu-blocklist", "--enable-webgl", "--disable-dev-shm-usage"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
await page.goto(url, { waitUntil: "networkidle", timeout: 90000 });
await page.waitForTimeout(Number(args.waitMs || 12000));
await page.screenshot({ path: out, fullPage: false });
await browser.close();
console.log(`wrote ${out} from ${url}`);

