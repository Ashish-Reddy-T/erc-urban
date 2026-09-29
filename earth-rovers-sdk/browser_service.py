import asyncio
import os
import time
from pyppeteer import launch
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "png")
QUALITY = float(os.getenv("IMAGE_QUALITY", "1.0"))
HAS_REAR_CAMERA = os.getenv("HAS_REAR_CAMERA", "False").lower() == "true"

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")


from pyppeteer.errors import ElementHandleError


class BrowserBusy(RuntimeError):
    """The headless browser is being (re)started; retry shortly."""


class BrowserService:
    def __init__(self):
        self.browser = None
        self.page = None
        # One launch at a time: /data, /v2/front and /control arrive
        # concurrently, and without this each launched its own headless Chrome
        # (three per recovery in the 2026-09-22 log) and they raced to
        # net::ERR_ABORTED.
        self._init_lock = asyncio.Lock()
        self.default_viewport = {"width": 3840, "height": 2160}

    async def initialize_browser(self):
        if self.browser:
            return
        if self._init_lock.locked():
            # Another request is already starting Chrome (~10-40 s). Do not
            # queue behind it: clients time out and retry at 10 Hz, and each
            # abandoned request kept its socket (221 in CLOSE_WAIT on
            # 2026-09-26 -> "Too many open files" at the 256 fd limit).
            raise BrowserBusy("headless browser is starting")
        async with self._init_lock:
            if self.browser:
                return
            await self._initialize_browser_locked()

    async def _initialize_browser_locked(self):
        if not self.browser:
            try:
                executable_path = os.getenv(
                    "CHROME_EXECUTABLE_PATH",
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                )
                self.browser = await launch(
                    executablePath=executable_path,
                    headless=True,
                    args=[
                        "--ignore-certificate-errors",
                        "--no-sandbox",
                        "--autoplay-policy=no-user-gesture-required",
                        "--use-fake-ui-for-media-stream",
                        f"--window-size={self.default_viewport['width']},{self.default_viewport['height']}",
                    ],
                )
                self.page = await self.browser.newPage()
                await self.page.setViewport(self.default_viewport)
                await self.page.setExtraHTTPHeaders(
                    {"Accept-Language": "en-US,en;q=0.9"}
                )
                await self.page.goto(
                    "http://127.0.0.1:8000/sdk", {"waitUntil": "networkidle2"}
                )
                await self.page.click("#join")
                await self.page.waitForSelector("video")
                await self.page.waitForSelector("#map")
                await self.page.setViewport(self.default_viewport)

                await self.page.waitFor(2000)

                call = f"""() => {{
                    window.initializeImageParams({{
                        imageFormat: "{FORMAT}",
                        imageQuality: {QUALITY}
                    }});
                }}"""
                await self.page.evaluate(call)
            except Exception as e:
                print(f"Error initializing browser: {e}")
                self.browser = None
                self.page = None
                await self.close_browser()
                raise

    async def take_screenshot(self, video_output_folder: str, elements: list):
        await self.initialize_browser()

        dimensions = await self.page.evaluate(
            """() => {
            return {
                width: Math.max(document.documentElement.scrollWidth, window.innerWidth),
                height: Math.max(document.documentElement.scrollHeight, window.innerHeight),
            }
        }"""
        )

        if (
            dimensions["width"] > self.default_viewport["width"]
            or dimensions["height"] > self.default_viewport["height"]
        ):
            await self.page.setViewport(dimensions)

        element_map = {"front": "#player-1000", "rear": "#player-1001", "map": "#map"}

        screenshots = {}
        for name in elements:
            if name in element_map:
                element_id = element_map[name]
                output_path = f"{video_output_folder}/{name}.png"
                element = await self.page.querySelector(element_id)
                if element:
                    start_time = time.time()  # Start time
                    await element.screenshot({"path": output_path})
                    end_time = time.time()  # End time
                    elapsed_time = (
                        end_time - start_time
                    ) * 1000  # Convert to milliseconds
                    print(f"Screenshot for {name} took {elapsed_time:.2f} ms")
                    screenshots[name] = output_path
                else:
                    print(f"Element {element_id} not found")
            else:
                print(f"Invalid element name: {name}")

        return screenshots

    async def data(self) -> dict:
        await self.initialize_browser()

        bot_data = await self.page.evaluate(
            """() => {
        return window.rtm_data;
        }"""
        )

        # Give RTM a moment on a freshly joined page
        if bot_data is None:
            await self.page.waitFor(3000)
            bot_data = await self.page.evaluate(
                """() => {
            return window.rtm_data;
            }"""
            )

        return bot_data

    async def rtm_status(self) -> dict:
        """RTM connection state as seen by the page (see static/basicRtm.js)."""
        await self.initialize_browser()
        return await self.page.evaluate(
            """() => ({state: window.rtm_state || null, reason: window.rtm_reason || null,
                      dead: !!window.rtm_dead, kicks: window.rtm_kicks || 0,
                      kick_times: window.rtm_kick_times || [],
                      peers: window.rtm_peers || {}, bot: $("#bot_uid").val()})"""
        )

    async def front(self) -> str:
        await self.initialize_browser()

        try:
            front_frame = await self.page.evaluate(
                """() => {
            return getLastBase64Frame(1000) || null;
            }"""
            )
        except ElementHandleError:
            # video not rendered yet (right after launch / a rejoin): the page
            # throws "source width is 0" -> answer 404, not a 500 + traceback
            return None

        return front_frame

    async def rear(self) -> str:
        await self.initialize_browser()

        try:
            rear_frame = await self.page.evaluate(
                """() => {
            return getLastBase64Frame(1001) || null;
            }"""
            )
        except ElementHandleError:
            # video not rendered yet (right after launch / a rejoin): the page
            # throws "source width is 0" -> answer 404, not a 500 + traceback
            return None

        return rear_frame

    async def send_message(self, message: dict):
        await self.initialize_browser()

        await self.page.evaluate(
            """(message) => {
                window.sendMessage(message);
            }""",
            message,
        )

    async def speak(self, audio_url: str):
        await self.initialize_browser()

        result = await self.page.evaluate(
            """async (audioUrl) => {
                return await window.playAudioToRover(audioUrl);
            }""",
            audio_url,
        )

        return result

    async def close_browser(self):
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None
