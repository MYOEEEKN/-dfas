// predictionLogic.js - SEROX AI Unified Consensus Core
// VERSION 9.0 - Single-File Architecture
// =================================================================

// --- SECTION 1: CORE UTILITY FUNCTIONS ---

function getBigSmallFromNumber(number) {
    if (number === undefined || number === null) return null;
    const num = parseInt(number);
    if (isNaN(num)) return null;
    return num >= 0 && num <= 4 ? 'SMALL' : num >= 5 && num <= 9 ? 'BIG' : null;
}

function calculateSMA(data, period) {
    if (!Array.isArray(data) || data.length < period || period <= 0) return null;
    const relevantData = data.slice(0, period);
    const sum = relevantData.reduce((a, b) => a + b, 0);
    return sum / period;
}

function calculateEMA(data, period) {
    if (!Array.isArray(data) || data.length < period || period <= 0) return null;
    const k = 2 / (period + 1);
    const chronologicalData = data.slice().reverse();
    let ema = calculateSMA(chronologicalData.slice(0, period).reverse(), period);
    if (ema === null) return null;
    for (let i = period; i < chronologicalData.length; i++) {
        ema = (chronologicalData[i] * k) + (ema * (1 - k));
    }
    return ema;
}

function calculateStdDev(data, period) {
    if (!Array.isArray(data) || data.length < period || period <= 0) return null;
    const relevantData = data.slice(0, period);
    if (relevantData.length < 2) return null;
    const mean = calculateSMA(relevantData, relevantData.length);
    const variance = relevantData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (relevantData.length - 1);
    return Math.sqrt(variance);
}

function calculateRSI(data, period) {
    if (!Array.isArray(data) || data.length < period + 1) return null;
    const chronologicalData = data.slice().reverse();
    let gains = 0, losses = 0;
    for (let i = 1; i <= period; i++) {
        const change = chronologicalData[i] - chronologicalData[i - 1];
        if (change > 0) gains += change; else losses += Math.abs(change);
    }
    let avgGain = gains / period;
    let avgLoss = losses / period;
    for (let i = period + 1; i < chronologicalData.length; i++) {
        const change = chronologicalData[i] - chronologicalData[i - 1];
        avgGain = (avgGain * (period - 1) + (change > 0 ? change : 0)) / period;
        avgLoss = (avgLoss * (period - 1) + (change < 0 ? Math.abs(change) : 0)) / period;
    }
    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
}


// --- SECTION 2: STATE MANAGEMENT & EVOLUTION ---

let systemState = {
    MIN_HISTORY: 100,
    BAD_TREND_THRESHOLD: 0.45,
    TARGET_ACCURACY: 0.54,
    EVOLUTION_RATE: 0.005,
    DEFENSIVE_MODE_ACTIVE: false,
};

let mlFeatureWeights = {
    rsi_strength: 1.5, rsi_is_overbought: -2.0, rsi_is_oversold: 2.0,
    macd_hist: 2.5, trend_strength_score: 3.0, bollinger_pct_reversal: -2.5,
    last_move: 0.5, market_sentiment: 1.0, stochastic_k: -1.8, rsi_trend_strength: 1.0,
};

function evolveSystemParameters(globalAccuracy) {
    if (globalAccuracy < systemState.TARGET_ACCURACY - 0.02) {
        systemState.BAD_TREND_THRESHOLD = Math.min(0.48, systemState.BAD_TREND_THRESHOLD + systemState.EVOLUTION_RATE);
    } else if (globalAccuracy > systemState.TARGET_ACCURACY + 0.02) {
        systemState.BAD_TREND_THRESHOLD = Math.max(0.42, systemState.BAD_TREND_THRESHOLD - systemState.EVOLUTION_RATE);
    }
}

function evolveMLWeights(history) {
    const learningRate = 0.01;
    const recentTrades = history.filter(p => p.mlFeatures && p.status).slice(0, 50);
    if (recentTrades.length < 20) return;

    let adjustments = Object.keys(mlFeatureWeights).reduce((acc, key) => ({ ...acc, [key]: 0 }), {});

    for (const trade of recentTrades) {
        const wasCorrect = trade.status === "Win";
        const wasBigPrediction = trade.lastPredictedOutcome === "BIG";
        for (const key in trade.mlFeatures) {
            if (mlFeatureWeights[key] === undefined) continue;
            const featureValue = trade.mlFeatures[key];
            const featureImpactIsBig = featureValue * mlFeatureWeights[key] > 0;
            const wasFeatureAlignedWithPrediction = wasBigPrediction ? featureImpactIsBig : !featureImpactIsBig;
            if (wasCorrect && wasFeatureAlignedWithPrediction) adjustments[key] += learningRate;
            else if (!wasCorrect && wasFeatureAlignedWithPrediction) adjustments[key] -= learningRate;
        }
    }
    for (const key in mlFeatureWeights) {
        mlFeatureWeights[key] = Math.max(0.1, Math.min(5.0, mlFeatureWeights[key] + adjustments[key]));
    }
}

function detectBadTrend(history) {
    const BAD_TREND_WINDOW = 30;
    if (!history || history.length < BAD_TREND_WINDOW) return false;
    const recentHistory = history.slice(0, BAD_TREND_WINDOW);
    const wins = recentHistory.filter(p => p.status === "Win").length;
    const losses = recentHistory.filter(p => p.status === "Loss").length;
    if (wins + losses < 15) return false;
    const accuracy = wins / (wins + losses);
    return accuracy < systemState.BAD_TREND_THRESHOLD;
}

function manageDefensiveMode(history) {
    if (detectBadTrend(history)) systemState.DEFENSIVE_MODE_ACTIVE = true;
    const last3 = history.slice(0, 3).map(p => p.status);
    if (systemState.DEFENSIVE_MODE_ACTIVE && last3.length === 3 && last3.every(s => s === 'Win')) {
        systemState.DEFENSIVE_MODE_ACTIVE = false;
    }
}

// --- SECTION 3: MARKET SENTIMENT SIMULATION ---
let marketEvents = [];
const NEWS_EVENT_PROBABILITY = 0.05;
const EVENT_IMPACT_DECAY_RATE = 0.90;

function updateMarketSentiment() {
    marketEvents = marketEvents.map(event => ({ ...event, impact: event.impact * EVENT_IMPACT_DECAY_RATE }))
        .filter(event => Math.abs(event.impact) > 0.05);
    if (Math.random() < NEWS_EVENT_PROBABILITY) {
        marketEvents.push({ type: 'News', impact: Math.random() - 0.5 });
    }
}

function getMarketSentimentFactor() {
    return marketEvents.reduce((acc, event) => acc + event.impact, 0);
}


// --- SECTION 4: ADVISORY PREDICTION MODELS ---

function analyzeRSITrend(history, rsiPeriod = 14, rsiMAPeriod = 9) {
    const numbers = history.map(p => p.actualNumber).filter(n => !isNaN(n));
    if (numbers.length < rsiPeriod + rsiMAPeriod) return null;
    const rsiValues = [];
    for (let i = rsiMAPeriod - 1; i >= 0; i--) {
        const rsi = calculateRSI(numbers.slice(i), rsiPeriod);
        if (rsi !== null) rsiValues.push(rsi); else return null;
    }
    const currentRSI = rsiValues[rsiValues.length - 1];
    const rsiMA = calculateSMA(rsiValues, rsiMAPeriod);
    if (currentRSI > rsiMA + 2) return { prediction: "BIG", source: "RSITrend" };
    if (currentRSI < rsiMA - 2) return { prediction: "SMALL", source: "RSITrend" };
    return null;
}

function analyzeStochastic(history, period = 14) {
    const numbers = history.map(p => p.actualNumber).filter(n => !isNaN(n)).slice(0, period);
    if (numbers.length < period) return null;
    const currentPrice = numbers[0];
    const lowestLow = Math.min(...numbers);
    const highestHigh = Math.max(...numbers);
    if (highestHigh === lowestLow) return null;
    const K = 100 * ((currentPrice - lowestLow) / (highestHigh - lowestLow));
    if (K > 85) return { prediction: "SMALL", source: "Stochastic" };
    if (K < 15) return { prediction: "BIG", source: "Stochastic" };
    return null;
}

function analyzeColorPatterns(history) {
    const outcomes = history.map(p => getBigSmallFromNumber(p.actual)).slice(0, 10).reverse();
    if (outcomes.length < 5) return null;
    const sequence = outcomes.join('');
    if (sequence.endsWith('BBBB')) return { prediction: 'BIG', source: 'Pattern:StreakCont' };
    if (sequence.endsWith('SSSS')) return { prediction: 'SMALL', source: 'Pattern:StreakCont' };
    if (sequence.endsWith('BBBBB')) return { prediction: 'SMALL', source: 'Pattern:StreakBreak' };
    if (sequence.endsWith('SSSSS')) return { prediction: 'BIG', source: 'Pattern:StreakBreak' };
    if (sequence.endsWith('BSBS')) return { prediction: 'BIG', source: 'Pattern:AltBreak' };
    if (sequence.endsWith('SBSB')) return { prediction: 'SMALL', source: 'Pattern:AltBreak' };
    return null;
}

function analyzeVolatilityBreakout(history, period = 20) {
    const numbers = history.map(p => p.actualNumber).filter(n => !isNaN(n));
    if (numbers.length < period * 2) return null;
    const recentVol = calculateStdDev(numbers.slice(0, period), period);
    const priorVol = calculateStdDev(numbers.slice(period, period * 2), period);
    if (recentVol === null || priorVol === null || priorVol === 0) return null;
    if (recentVol > priorVol * 1.8) {
        return { prediction: numbers[0] > numbers[1] ? "BIG" : "SMALL", source: "Volatility" };
    }
    return null;
}

function analyzePriceAction(history) {
    const numbers = history.map(p => p.actualNumber).filter(n => !isNaN(n)).slice(0, 5);
    if (numbers.length < 5) return null;
    const [p0, p1, p2, p3] = numbers;
    if (p0 > p2 && p1 > p3) return { prediction: 'BIG', source: 'PriceAction' };
    if (p0 < p2 && p1 < p3) return { prediction: 'SMALL', source: 'PriceAction' };
    return null;
}

function analyzeMeanReversion(history, period = 20) {
    const numbers = history.map(p => p.actualNumber).filter(n => !isNaN(n)).slice(0, period);
    if (numbers.length < period) return null;
    const sma = calculateSMA(numbers, period);
    const stdDev = calculateStdDev(numbers, period);
    if (sma === null || stdDev === null) return null;
    const zScore = (numbers[0] - sma) / stdDev;
    if (zScore > 1.5) return { prediction: 'SMALL', source: 'MeanReversion' };
    if (zScore < -1.5) return { prediction: 'BIG', source: 'MeanReversion' };
    return null;
}

function runAdvisoryModels(history, primaryPrediction) {
    const models = [
        analyzeRSITrend(history), analyzeStochastic(history),
        analyzeColorPatterns(history), analyzeVolatilityBreakout(history),
        analyzePriceAction(history), analyzeMeanReversion(history)
    ];
    const advisorySignals = models.filter(m => m !== null);
    let agreeingModels = advisorySignals.filter(m => m.prediction === primaryPrediction).length;
    const totalAdvisors = advisorySignals.length;
    const consensusScore = totalAdvisors > 0 ? (agreeingModels / totalAdvisors) : 0.5;
    return { advisorySignals, consensusScore, agreeingModels, totalAdvisors };
}

// --- SECTION 5: PRIMARY LEARNING MODEL ---

function getTrendContext(history, longMALookback = 20) {
    const numbers = history.map(entry => entry.actualNumber).filter(n => !isNaN(n));
    if (numbers.length < longMALookback) return { strength: "UNKNOWN", direction: "NONE" };
    const shortMA = calculateEMA(numbers, 5);
    const mediumMA = calculateEMA(numbers, 10);
    const longMA = calculateEMA(numbers, longMALookback);
    if (shortMA === null || mediumMA === null || longMA === null) return { strength: "UNKNOWN", direction: "NONE" };
    let direction = "NONE", strength = "WEAK";
    if (shortMA > mediumMA && mediumMA > longMA) { direction = "BIG"; strength = "STRONG"; }
    else if (shortMA < mediumMA && mediumMA < longMA) { direction = "SMALL"; strength = "STRONG"; }
    else { strength = "RANGING"; }
    return { strength, direction };
}

function createFeatureSetForML(history) {
    const numbers = history.map(e => e.actualNumber).filter(n => !isNaN(n));
    if (numbers.length < systemState.MIN_HISTORY) return null;
    const trendContext = getTrendContext(history);
    const rsiValue = calculateRSI(numbers, 14);
    const macdLine = calculateEMA(numbers, 12) - calculateEMA(numbers, 26);
    const signalLine = calculateEMA(numbers.map((_, i) => calculateEMA(numbers.slice(i), 12) - calculateEMA(numbers.slice(i), 26)).filter(n => n !== null), 9);
    return {
        rsi_strength: rsiValue ? (rsiValue - 50) / 50 : 0,
        rsi_is_overbought: rsiValue && rsiValue > 70 ? 1 : 0,
        rsi_is_oversold: rsiValue && rsiValue < 30 ? -1 : 0,
        macd_hist: macdLine && signalLine ? macdLine - signalLine : 0,
        trend_strength_score: trendContext.strength === 'STRONG' ? (trendContext.direction.includes('BIG') ? 1 : -1) : 0,
        last_move: numbers[0] > numbers[1] ? 1 : -1,
    };
}

function analyzeUnifiedMLModel(features) {
    if (!features) return null;
    let bigScore = 0, smallScore = 0;
    for (const key in features) {
        if (mlFeatureWeights[key] !== undefined) {
            const weight = mlFeatureWeights[key];
            const featureValue = features[key];
            if (weight > 0) {
                if (featureValue > 0) bigScore += featureValue * weight;
                else smallScore += Math.abs(featureValue * weight);
            } else {
                if (featureValue > 0) smallScore += featureValue * Math.abs(weight);
                else bigScore += Math.abs(featureValue * Math.abs(weight));
            }
        }
    }
    const totalScore = bigScore + smallScore;
    if (totalScore === 0) return null;
    const confidence = Math.abs(bigScore - smallScore) / totalScore;
    const prediction = bigScore > smallScore ? "BIG" : "SMALL";
    return { prediction, confidence, source: "LearningML" };
}


// --- SECTION 6: MAIN PREDICTION ORCHESTRATOR ---

function ultraAIPredict(currentSharedHistory, sharedStatsPayload) {
    const confirmedHistory = currentSharedHistory.filter(p => p && p.actual !== null && p.actualNumber !== undefined);

    if (confirmedHistory.length < systemState.MIN_HISTORY) {
        return { finalDecision: Math.random() > 0.5 ? "BIG" : "SMALL", confidenceLevel: 1, source: "ConsensusCore-v9.0", systemHealth: "INSUFFICIENT_HISTORY" };
    }

    if (confirmedHistory.length % 5 === 0) {
        if (sharedStatsPayload.longTermGlobalAccuracy) {
            evolveSystemParameters(sharedStatsPayload.longTermGlobalAccuracy);
        }
        evolveMLWeights(confirmedHistory);
        updateMarketSentiment();
    }

    const lastResult = sharedStatsPayload?.lastActualOutcome ? {
        status: getBigSmallFromNumber(sharedStatsPayload.lastActualOutcome) === sharedStatsPayload.lastPredictedOutcome ? "Win" : "Loss"
    } : null;

    if (lastResult) {
        manageDefensiveMode(confirmedHistory);
    }

    const mlFeatures = createFeatureSetForML(confirmedHistory);
    const primaryModel = analyzeUnifiedMLModel(mlFeatures);

    if (!primaryModel) {
        return { finalDecision: Math.random() > 0.5 ? "BIG" : "SMALL", confidenceLevel: 0, source: "ConsensusCore-v9.0", systemHealth: "MODEL_UNCERTAIN" };
    }

    const { advisorySignals, consensusScore, agreeingModels, totalAdvisors } = runAdvisoryModels(confirmedHistory, primaryModel.prediction);

    let finalConfidence = primaryModel.confidence * (0.6 + (consensusScore * 0.4));
    if (systemState.DEFENSIVE_MODE_ACTIVE) {
        finalConfidence *= 0.7;
    }

    let confidenceLevel = (finalConfidence > 0.55) ? 1 : 0;
    if (systemState.DEFENSIVE_MODE_ACTIVE) {
        confidenceLevel = 0;
    }

    const output = {
        finalDecision: primaryModel.prediction,
        finalConfidence,
        confidenceLevel,
        overallLogic: "ConsensusCore-v9.0",
        source: `ML+${agreeingModels}/${totalAdvisors}_Advisors`,
        systemHealth: systemState.DEFENSIVE_MODE_ACTIVE ? `DEFENSIVE_MODE` : "OK",
    };

    Object.assign(sharedStatsPayload, {
        ...output,
        lastPredictedOutcome: output.finalDecision,
        mlFeatures: mlFeatures,
        status: lastResult ? lastResult.status : 'Pending'
    });
    
    return output;
}

// --- SECTION 7: EXPORTS FOR NODE.JS SERVER ---

module.exports = {
    ultraAIPredict,
    getBigSmallFromNumber
};
