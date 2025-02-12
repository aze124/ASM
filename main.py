import discord
from discord.ext import commands
import asyncio
import random
from playwright.async_api import async_playwright
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque, defaultdict
from datetime import datetime, timezone
import cv2
import pytesseract
from PIL import Image
from io import BytesIO
import logging
import os

# --- Configuration Section ---
CURRENT_TIME = "2025-02-12 02:59:59"
CURRENT_USER = "aze124"
CHANNEL_ID = 1311005802877948050
DISCORD_BOT_TOKEN = "MTMxMTAwNTg1NTQ4Njg0MDg5NQ.G8NM0y.dzDfThwPFIQqYy0wS9MizpWJ9EpohT1hrsTee0"
WEBSITE_URL = "https://bloxluck.com"
RATE_LIMITER_DELAY = 0.1
MAX_GAME_HISTORY = 500
MAX_TIME_HISTORY = 300
MAX_PREDICTION_HISTORY = 100
MAX_ACTUAL_RESULTS = 100
MAX_LAST_PREDICTIONS = 5
RECENT_HISTORY_SIZE = 5
MIN_CONFIDENCE_LEVEL = 0.0
RF_ESTIMATORS = 200
GB_ESTIMATORS = 150
RANDOM_SEED = 42
OPENCV_CHANGE_THRESHOLD = 50000
TESSERACT_OEM = 3
TESSERACT_PSM = 6
TESSERACT_CONFIG = f'--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- GameState Class Definition ---
class GameState:
    def __init__(self):
        logging.info("Initializing GameState object...")
        self.monitor_active = True
        self.last_game_result = None
        self.game_history = deque(maxlen=MAX_GAME_HISTORY)
        self.time_history = deque(maxlen=MAX_TIME_HISTORY)
        self.prediction_history = deque(maxlen=MAX_PREDICTION_HISTORY)
        self.actual_results = deque(maxlen=MAX_ACTUAL_RESULTS)
        self.last_prediction = None
        self.prediction_sent = False
        self.wrong_predictions_count = 0
        self.current_streak = 0
        self.streak_color = None
        self.current_game_result = "Unknown"
        self.baseline_image = None
        self.tesseract_config = TESSERACT_CONFIG
        self.last_predictions = deque(maxlen=MAX_LAST_PREDICTIONS)

        self.player_history = defaultdict(list)
        self.player_predictions = defaultdict(list)
        self.player_timestamps = defaultdict(list)
        self.player_streaks = defaultdict(int)

        self.rf_model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_SEED)
        self.gb_model = GradientBoostingClassifier(n_estimators=GB_ESTIMATORS, random_state=RANDOM_SEED)
        self.voting_model = VotingClassifier(
            estimators=[('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier())],
            voting='soft'
        )
        self.scaler = StandardScaler()
        self.trained = False

        self.min_confidence = MIN_CONFIDENCE_LEVEL
        self.streak_threshold = 4
        self.blue_win_rate = 0.5
        self.adaptive_confidence_multiplier = 1.0

        self.patterns = {
            "BBBBB": ("Red", 0.95),
            "RRRRR": ("Blue", 0.95),
            "BBBB": ("Red", 0.90),
            "RRRR": ("Blue", 0.90),
            "BBB": ("Red", 0.80),
            "RRR": ("Blue", 0.80),
            "BRBR": ("Red", 0.75),
            "RBRB": ("Blue", 0.75),
            "BBRR": ("Red", 0.70),
            "RRBB": ("Blue", 0.70),
            "BRB": ("Red", 0.65),
            "RBR": ("Blue", 0.65),
            "BB": ("Red", 0.60),
            "RR": ("Blue", 0.60),
        }
        logging.info("GameState object initialized successfully.")

    def track_player_game(self, player, result, timestamp):
        try:
            self.player_history[player].append(result)
            self.player_timestamps[player].append(timestamp)
            if len(self.player_history[player]) > 1:
                if self.player_history[player][-1] == self.player_history[player][-2]:
                    self.player_streaks[player] += 1
                else:
                    self.player_streaks[player] = 1
            else:
                self.player_streaks[player] = 1

            if len(self.player_history[player]) > 100:
                self.player_history[player].pop(0)
                self.player_timestamps[player].pop(0)
            logging.debug(f"Tracked game for player: {player}, Result: {result}")
        except Exception as e:
            logging.error(f"Error tracking game for player {player}: {e}", exc_info=True)

    def get_smart_fallback_prediction(self):
        try:
            if not self.game_history:
                if self.current_game_result and self.current_game_result != "Unknown":
                    prediction = "Red" if self.current_game_result == "Blue" else "Blue"
                    confidence = 50.0
                else:
                    prediction = random.choice(["Blue", "Red"])
                    confidence = 50.0
            else:
                last_result = list(self.game_history)[-1]
                streak = 1
                for i in range(len(self.game_history)-2, -1, -1):
                    if self.game_history[i] == last_result:
                        streak += 1
                    else:
                        break

                if streak >= 5:
                    opposite = "Red" if last_result == "Blue" else "Blue"
                    return opposite, 95.0
                elif streak >= 4:
                    opposite = "Red" if last_result == "Blue" else "Blue"
                    return opposite, 90.0
                elif streak >= 3:
                    opposite = "Red" if last_result == "Blue" else "Blue"
                    return opposite, 80.0
                elif streak >= 2:
                    opposite = "Red" if last_result == "Blue" else "Blue"
                    return opposite, 70.0
                else:
                    opposite = "Red" if last_result == "Blue" else "Blue"
                    return opposite, 60.0

            if len(self.last_predictions) >= 3:
                if all(p == "Blue" for p in self.last_predictions):
                    return "Red", 75.0
                if all(p == "Red" for p in self.last_predictions):
                    return "Blue", 75.0

            prediction = random.choice(["Blue", "Red"])
            confidence = 50.0
            return prediction, confidence
        except Exception as e:
            logging.error(f"Error in get_smart_fallback_prediction: {e}", exc_info=True)
            return random.choice(["Blue", "Red"]), 50.0

    def get_pattern_prediction(self):
        try:
            if len(self.game_history) < 2:
                if not self.game_history:
                    if self.current_game_result and self.current_game_result != "Unknown":
                        prediction = "Red" if self.current_game_result == "Blue" else "Blue"
                        confidence = 60.0
                        return prediction, confidence
                    else:
                        prediction = random.choice(["Blue", "Red"])
                        confidence = 50.0
                        return prediction, confidence
                last_result = list(self.game_history)[-1]
                opposite = "Red" if last_result == "Blue" else "Blue"
                confidence = 60.0
                return opposite, confidence

            recent = list(self.game_history)[-RECENT_HISTORY_SIZE:]
            pattern = ''.join(['B' if g == "Blue" else 'R' for g in recent])

            for pat, (pred, conf) in self.patterns.items():
                if pattern.endswith(pat):
                    if self.wrong_predictions_count >= 3:
                        conf *= 0.85
                    confidence = conf * self.adaptive_confidence_multiplier
                    return pred, confidence

            streak = 1
            last_color = recent[-1]
            for i in range(len(recent)-2, -1, -1):
                if recent[i] == last_color:
                    streak += 1
                else:
                    break

            if streak >= 4:
                opposite = "Red" if last_color == "Blue" else "Blue"
                return opposite, 90.0
            elif streak >= 2:
                opposite = "Red" if last_color == "Blue" else "Blue"
                return opposite, 70.0

            if len(recent) >= 4:
                if recent[-4:] == ["Blue", "Red", "Blue", "Red"]:
                    return "Blue", 75.0
                if recent[-4:] == ["Red", "Blue", "Red", "Blue"]:
                    return "Red", 75.0

            return self.get_smart_fallback_prediction()
        except Exception as e:
            logging.error(f"Error in get_pattern_prediction: {e}", exc_info=True)
            return self.get_smart_fallback_prediction()

    def get_player_prediction(self, player):
        try:
            if player not in self.player_history or not self.player_history[player]:
                return self.get_smart_fallback_prediction()

            history = self.player_history[player]
            recent = history[-10:] if len(history) >= 10 else history

            if self.player_streaks[player] >= 4:
                last_color = history[-1]
                opposite = "Red" if last_color == "Blue" else "Blue"
                return opposite, 90.0

            elif self.player_streaks[player] >= 2:
                last_color = history[-1]
                opposite = "Red" if last_color == "Blue" else "Blue"
                return opposite, 70.0

            blue_count = sum(1 for g in recent if g == "Blue")
            red_count = len(recent) - blue_count
            if blue_count > red_count * 1.5:
                return "Red", 80.0
            if red_count > blue_count * 1.5:
                return "Blue", 80.0

            pattern_pred, pattern_conf = self.get_pattern_prediction()
            return pattern_pred, pattern_conf * 1.1
        except Exception as e:
            logging.error(f"Error in get_player_prediction for player {player}: {e}", exc_info=True)
            return self.get_smart_fallback_prediction()

    def calculate_accuracy(self):
        try:
            if not self.prediction_history or not self.actual_results:
                logging.warning("No prediction history or actual results to calculate accuracy.")
                return 0.0
            correct = sum(1 for p, a in zip(self.prediction_history, self.actual_results) if p == a)
            accuracy = (correct / len(self.prediction_history)) * 100
            logging.debug(f"Calculated accuracy: {accuracy:.2f}%")
            return accuracy
        except Exception as e:
            logging.error(f"Error in calculate_accuracy: {e}", exc_info=True)
            return 0.0

    def update_adaptive_confidence(self, correct_prediction):
        try:
            if len(self.prediction_history) < 10:
                logging.info("Not enough predictions to update adaptive confidence.")
                return
            recent_predictions = list(self.prediction_history)[-10:]
            recent_actuals = list(self.actual_results)[-10:]
            correct_count = sum(1 for p, a in zip(recent_predictions, recent_actuals) if p == a)
            accuracy = correct_count / 10
            if accuracy >= 0.7:
                self.adaptive_confidence_multiplier = min(1.1, self.adaptive_confidence_multiplier + 0.02)
                logging.info(f"Increased adaptive confidence multiplier to: {self.adaptive_confidence_multiplier:.2f}")
            else:
                self.adaptive_confidence_multiplier = max(0.9, self.adaptive_confidence_multiplier - 0.03)
                logging.info(f"Decreased adaptive confidence multiplier to: {self.adaptive_confidence_multiplier:.2f}")
        except Exception as e:
            logging.error(f"Error in update_adaptive_confidence: {e}", exc_info=True)

    async def real_time_update(self):
        try:
            async with async_playwright() as p:
                logging.info("Launching browser for real-time updates.")
                browser = await p.chromium.launch(headless=True, args=['--start-maximized'])
                page = await browser.new_page()
                logging.info(f"Navigating to {WEBSITE_URL}")
                await page.goto(WEBSITE_URL, wait_until='domcontentloaded', timeout=30000)
                await page.set_viewport_size({"width": 1920, "height": 1080})
                logging.info("Taking initial screenshot.")
                initial_screenshot = await page.screenshot(full_page=True)
                pil_image = Image.open(BytesIO(initial_screenshot))
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                self.baseline_image = cv_image.copy()
                logging.info("Initial screenshot taken and stored.")

                while self.monitor_active:
                    try:
                        new_screenshot = await page.screenshot(full_page=True)
                        new_pil_image = Image.open(BytesIO(new_screenshot))
                        new_cv_image = cv2.cvtColor(np.array(new_pil_image), cv2.COLOR_RGB2BGR)
                        difference_image = cv2.absdiff(self.baseline_image, new_cv_image)
                        gray_difference = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)
                        _, thresholded = cv2.threshold(gray_difference, 30, 255, cv2.THRESH_BINARY)
                        if cv2.countNonZero(thresholded) > OPENCV_CHANGE_THRESHOLD:
                            logging.info("Significant Change Detected on Whole Page")
                            text = pytesseract.image_to_string(new_cv_image, config=self.tesseract_config).strip()
                            if "Blue" in text or "Red" in text:
                                result = "Blue" if "Blue" in text else "Red"
                                logging.info(f"OCR Result: {result}")
                                self.current_game_result = result
                                if result != self.last_game_result:
                                    self.last_game_result = result
                                    self.game_history.append(result)
                                    self.time_history.append(datetime.now(timezone.utc))
                                    train_models()
                                    logging.info(f"New Game Result: {result}, Retraining Models")
                                    self.baseline_image = new_cv_image.copy()
                            else:
                                logging.warning("OCR failed to reliably read result.")
                                if self.last_game_result and self.last_game_result != "Unknown":
                                    prediction = "Red" if self.last_game_result == "Blue" else "Blue"
                                    self.current_game_result = prediction
                                else:
                                    self.current_game_result = "Unknown"
                        await asyncio.sleep(RATE_LIMITER_DELAY)
                    except Exception as e:
                        logging.error(f"Error during real-time OpenCV update: {e}", exc_info=True)
                        if self.last_game_result and self.last_game_result != "Unknown":
                            self.current_game_result = self.last_game_result
                        else:
                            self.current_game_result = "Unknown"
                await browser.close()
        except Exception as e:
            logging.error(f"Error launching browser for real-time OpenCV updates: {e}", exc_info=True)
            self.current_game_result = "Unknown"

    async def scrape_current_result(self):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(WEBSITE_URL, wait_until='domcontentloaded', timeout=30000)
                current_result = await page.evaluate('''() => {
                    const resultElement = document.querySelector('.game-result');
                    if (!resultElement) return null;
                    const result = resultElement.innerText.trim();
                    return ['Blue', 'Red'].includes(result) ? result : null;
                }''')
                await browser.close()
                return current_result
        except Exception as e:
            logging.error(f"Error scraping current result: {e}", exc_info=True)
            return None

# --- Global Game State Instance ---
game_state = GameState()

# --- Helper Functions ---
def train_models():
    if len(game_state.game_history) < 20:
        logging.warning("Not enough game history to train models.")
        return False
    try:
        numeric_history = [1 if g == "Blue" else 0 for g in game_state.game_history]
        X = np.array(numeric_history[:-1]).reshape(-1, 1)
        y = np.array(numeric_history[1:])
        X_scaled = game_state.scaler.fit_transform(X)
        game_state.rf_model.fit(X_scaled, y)
        game_state.gb_model.fit(X_scaled, y)
        game_state.voting_model.fit(X_scaled, y)
        game_state.trained = True
        logging.info("Machine learning models trained successfully.")
        return True
    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        return False

def predict_next_game():
    pattern_pred, pattern_conf = game_state.get_pattern_prediction()
    if game_state.trained and len(game_state.game_history) >= 20:
        try:
            last_result = 1 if game_state.game_history[-1] == "Blue" else 0
            X_scaled = game_state.scaler.transform([[last_result]])
            rf_proba = game_state.rf_model.predict_proba(X_scaled)[0]
            gb_proba = game_state.gb_model.predict_proba(X_scaled)[0]
            vt_proba = game_state.voting_model.predict_proba(X_scaled)[0]
            ensemble_proba = (rf_proba * 0.4 + gb_proba * 0.3 + vt_proba * 0.3)
            ml_pred = "Blue" if ensemble_proba[1] > 0.5 else "Red"
            ml_conf = max(ensemble_proba) * 100

            if ml_pred == pattern_pred:
                final_pred = ml_pred
                final_conf = round((ml_conf * 0.3 + pattern_conf * 0.7), 2)
            elif ml_conf > pattern_conf:
                final_pred = ml_pred
                final_conf = round(ml_conf * 0.5 + pattern_conf * 0.5, 2)
            else:
                final_pred = pattern_pred
                final_conf = round(pattern_conf * 0.6 + ml_conf * 0.4, 2)

            final_conf *= game_state.adaptive_confidence_multiplier
            final_conf = min(100.0, final_conf)
            return final_pred, round(final_conf, 2)
        except Exception as e:
            logging.error(f"Prediction error: {e}", exc_info=True)
            return pattern_pred, pattern_conf
    return pattern_pred, pattern_conf

# --- Discord Bot Events and Commands ---
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user.name}')
    logging.info(f'Current time: {CURRENT_TIME}')
    logging.info(f'User: {CURRENT_USER}')
    bot.loop.create_task(website_monitor())

@bot.command(name='p')
async def player_prediction(ctx, *, username=None):
    current_result = game_state.current_game_result
    if username:
        if username not in game_state.player_history:
            if current_result and current_result != "Unknown":
                prediction = "Red" if current_result == "Blue" else "Blue"
                confidence = 99.0
            else:
                prediction, confidence = "Blue", 50.0
            embed = discord.Embed(
                title=f"👤 Player Analysis: {username} (No History)",
                description=f"**Current Game Result**: {current_result}\n"
                            f"**Next Prediction**: ||{prediction}|| ({confidence:.1f}% confidence)",
                color=0x3498db if prediction == "Blue" else 0xe74c3c,
                timestamp=datetime.now(timezone.utc)
            )
            await ctx.send(embed=embed)
            return

        prediction, confidence = game_state.get_player_prediction(username)
        history = game_state.player_history[username]
        timestamps = game_state.player_timestamps[username]

        embed = discord.Embed(
            title=f"👤 Player Analysis: {username}",
            description=f"**Current Game Result**: {current_result}\n"
                        f"**Next Prediction**: ||{prediction}|| ({confidence:.1f}% confidence)",
            color=0x3498db if prediction == "Blue" else 0xe74c3c,
            timestamp=datetime.now(timezone.utc)
        )

        recent_games = list(zip(history[-10:], timestamps[-10:]))
        recent_text = "\n".join(
            f"{i+1}. {game} at {ts.strftime('%H:%M:%S')}"
            for i, (game, ts) in enumerate(recent_games)
        )
        embed.add_field(
            name="🎮 Recent Games",
            value=recent_text if recent_text else "No recent games",
            inline=False
        )

        total_games = len(history)
        blue_count = sum(1 for g in history if g == "Blue")
        red_count = total_games - blue_count
        stats_text = (
            f"Total Games: {total_games}\n"
            f"Blue Rate: {(blue_count/total_games)*100:.1f}%,\n"
            f"Red Rate: {(red_count/total_games)*100:.1f}%,\n"
            f"Current Streak: {game_state.player_streaks[username]}"
        )
        embed.add_field(
            name="📊 Statistics",
            value=stats_text,
            inline=False
        )

        pattern_pred, pattern_conf = game_state.get_pattern_prediction()
        embed.add_field(
            name="🎲 Pattern Analysis",
            value=f"Pattern suggests: ||{pattern_pred}|| ({pattern_conf:.1f}%)",
            inline=False
        )

        await ctx.send(embed=embed)
    else:
        prediction, confidence = predict_next_game()
        embed = discord.Embed(
            title=f"🎲 General Prediction",
            description=f"**Current Game Result**: {current_result}\n"
                        f"**Next Prediction**: ||{prediction}|| ({confidence:.1f}% confidence)",
            color=0x3498db if prediction == "Blue" else 0xe74c3c,
            timestamp=datetime.now(timezone.utc)
        )
        await ctx.send(embed=embed)

@bot.command(name='force')
async def force_predict(ctx):
    current_result = game_state.current_game_result
    if current_result and current_result != "Unknown":
        prediction = "Red" if current_result == "Blue" else "Blue"
        confidence = 99.0
    else:
        predictions = []
        for _ in range(3):
            prediction, confidence = predict_next_game()
            predictions.append((prediction, confidence))

        # Sort predictions by confidence
        sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        # Get the most confident prediction
        best_prediction, best_confidence = sorted_predictions[0]

        # Check for majority
        blue_count = sum(1 for pred, _ in predictions if pred == "Blue")
        red_count = sum(1 for pred, _ in predictions if pred == "Red")

        if blue_count > red_count:
            conclusion = "Blue"
        elif red_count > blue_count:
            conclusion = "Red"
        else:
            conclusion = best_prediction

        embed = discord.Embed(
            title="🔮 Forced Prediction",
            description=f"**Current Game Result**: {current_result}\n"
                        f"**Predictions**: ||{'||, ||'.join(f'{pred} ({conf:.1f}%)' for pred, conf in predictions)}||\n"
                        f"**Conclusion**: ||{conclusion}|| ({best_confidence:.1f}% confidence)",
            color=0x3498db if conclusion == "Blue" else 0xe74c3c,
            timestamp=datetime.now(timezone.utc)
        )
        await ctx.send(embed=embed)

# --- Website Monitoring Task ---
async def website_monitor():
    while True:
        await game_state.real_time_update()
        await asyncio.sleep(RATE_LIMITER_DELAY)

# --- Run the bot ---
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
