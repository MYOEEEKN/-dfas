// index.js - SEROX AI Unified Server
// VERSION 8.0 - Single Deployable Architecture
// =================================================================
import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

// --- Core Application Imports ---
import { ultraAIPredict } from './main.js';
import { getBigSmallFromNumber } from './utils.js';

// --- Server Setup ---
const app = express();
const PORT = process.env.PORT || 3000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Middleware ---
app.use(cors({ origin: '*' }));
app.use(express.json());
// Serve static files (index.html) from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));


// --- In-Memory State ---
let lastProcessedPeriod = null;
let inMemoryHistory = [];
let inMemorySharedStats = {};
let inMemoryCurrentPrediction = null;
const MAX_HISTORY_LENGTH = 150;


// --- API Endpoint ---
app.post('/predict', async (req, res) => {
    const { gameResult } = req.body;
    if (!gameResult || !gameResult.issueNumber) {
        return res.status(400).json({ success: false, message: "Missing gameResult in request body." });
    }

    try {
        const endedPeriodFull = gameResult.issueNumber.trim();
        if (endedPeriodFull === lastProcessedPeriod) {
            return res.json({
                success: true,
                message: "Period already processed.",
                currentPrediction: inMemoryCurrentPrediction,
                history: inMemoryHistory.slice(0, 50)
            });
        }

        const actualNumber = Number(gameResult.number);
        const actualResultType = getBigSmallFromNumber(actualNumber);
        const previousSharedPrediction = inMemoryCurrentPrediction;

        if (previousSharedPrediction && previousSharedPrediction.period === endedPeriodFull) {
            let statusOfPreviousPrediction = 'Loss';
            if (previousSharedPrediction.prediction === 'DEFENSIVE_MODE' || previousSharedPrediction.prediction === 'COOLDOWN') {
                statusOfPreviousPrediction = 'Cooldown';
            } else if (actualResultType === previousSharedPrediction.prediction) {
                statusOfPreviousPrediction = 'Win';
            }
            const historyEntryToUpdate = inMemoryHistory.find(entry => entry.period === endedPeriodFull);
            if(historyEntryToUpdate) {
                historyEntryToUpdate.status = statusOfPreviousPrediction;
            }
            inMemorySharedStats.lastActualOutcome = actualNumber;
            inMemorySharedStats.lastPredictedOutcome = previousSharedPrediction.prediction;
            inMemorySharedStats.lastConfidenceLevel = previousSharedPrediction.confidenceLevel;
        }

        inMemoryHistory.unshift({
            period: endedPeriodFull,
            actual: actualNumber,
            actualNumber: actualNumber,
            resultType: actualResultType,
            status: 'Pending',
            timestamp: Date.now()
        });

        if (inMemoryHistory.length > MAX_HISTORY_LENGTH) {
            inMemoryHistory.pop();
        }

        lastProcessedPeriod = endedPeriodFull;

        const aiDecision = ultraAIPredict(inMemoryHistory, inMemorySharedStats);

        const nextPeriodToPredictFull = (BigInt(endedPeriodFull) + 1n).toString();
        inMemoryCurrentPrediction = {
            period: nextPeriodToPredictFull,
            prediction: aiDecision.finalDecision,
            confidence: aiDecision.finalConfidence ? Math.round(aiDecision.finalConfidence * 100) : 50,
            confidenceLevel: aiDecision.confidenceLevel,
            overallLogic: aiDecision.overallLogic,
            source: aiDecision.source,
            systemHealth: aiDecision.systemHealth,
            timestamp: Date.now()
        };

        res.json({
            success: true,
            message: "Prediction cycle complete.",
            currentPrediction: inMemoryCurrentPrediction,
            history: inMemoryHistory.slice(0, 50)
        });

    } catch (error) {
        console.error("Error in /predict endpoint:", error);
        res.status(500).json({ success: false, message: error.message || "Internal server error." });
    }
});

// --- Root Endpoint to Serve the UI ---
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});


// --- Start Server ---
function startServer() {
    app.listen(PORT, () => {
        console.log(`SEROX AI Unified Server running on port ${PORT}`);
    });
}

startServer();
