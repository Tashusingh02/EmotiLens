const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Store chat instances (in production, use a proper session store)
const chatInstances = new Map();

// Store emotion frequency for book recommendations (keeping for potential future use)
// Format: { emotion: count }
const emotionFrequency = {
    'neutral': 0,
    'joy': 0,
    'sadness': 0,
    'fear': 0,
    'anger': 0
};

// Store most recent emotion for book recommendations
let mostRecentEmotion = 'neutral';

// Helper function to get emotion bucket from emotion name
function getEmotionBucket(emotion) {
    const emotionLower = emotion.toLowerCase();
    if (['joy', 'excitement', 'amusement', 'gratitude', 'love', 'optimism', 'pride', 'relief', 'admiration', 'approval', 'caring'].includes(emotionLower)) {
        return 'joy';
    } else if (['sadness', 'grief', 'disappointment', 'remorse'].includes(emotionLower)) {
        return 'sadness';
    } else if (['fear', 'nervousness'].includes(emotionLower)) {
        return 'fear';
    } else if (['anger', 'annoyance', 'disapproval', 'disgust'].includes(emotionLower)) {
        return 'anger';
    }
    return 'neutral';
}

// Helper function to update emotion frequency and most recent emotion
function updateEmotionFrequency(emotion) {
    const bucket = getEmotionBucket(emotion);

    // Update frequency (keeping for potential future use)
    if (emotionFrequency.hasOwnProperty(bucket)) {
        emotionFrequency[bucket]++;
    } else {
        emotionFrequency['neutral']++;
    }

    // Update most recent emotion
    mostRecentEmotion = bucket;
    console.log(`Most recent emotion updated to: ${mostRecentEmotion}`);
}

// Helper function to get most recent emotion (for book recommendations)
function getMostRecentEmotion() {
    return mostRecentEmotion;
}

// Helper function to call Python hybrid_chat
function callHybridChat(sessionId, message, isNewChat = false) {
    return new Promise((resolve, reject) => {
        const pythonScript = path.join(__dirname, 'hybrid_chat_api.py');

        // Try python3 first, then python
        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

        const pythonProcess = spawn(pythonCmd, [
            pythonScript,
            sessionId,
            message,
            isNewChat ? 'true' : 'false'
        ], {
            cwd: path.join(__dirname, '..'), // Set working directory to project root
            env: { ...process.env, PYTHONUNBUFFERED: '1' }, // Unbuffered output
            timeout: 360000 // 6 minutes timeout for the entire Python process
        });

        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data) => {
            const text = data.toString();
            output += text;
            // Also log to console for debugging
            console.log('Python stdout:', text);
        });

        pythonProcess.stderr.on('data', (data) => {
            const text = data.toString();
            errorOutput += text;
            // Log to console for debugging
            console.error('Python stderr:', text);
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                // Try to parse error output as JSON
                let errorMessage = errorOutput || output;
                try {
                    const errorJson = JSON.parse(errorMessage.trim());
                    if (errorJson.error) {
                        reject(new Error(errorJson.error));
                    } else {
                        reject(new Error(`Python process exited with code ${code}: ${errorMessage}`));
                    }
                } catch (e) {
                    reject(new Error(`Python process exited with code ${code}: ${errorMessage}`));
                }
                return;
            }

            try {
                const result = JSON.parse(output.trim());
                resolve(result);
            } catch (e) {
                reject(new Error(`Failed to parse Python output: ${output}`));
            }
        });

        pythonProcess.on('error', (error) => {
            const errorMsg = error.code === 'ENOENT'
                ? `Python not found. Please make sure Python is installed and in your PATH. Tried: ${pythonCmd}`
                : `Failed to start Python process: ${error.message}`;
            reject(new Error(errorMsg));
        });
    });
}

// API Routes
app.post('/api/chat', async (req, res) => {
    try {
        const { message, text, sessionId = 'default', reset = false, advice_given = false, history = [] } = req.body;
        const userMsg = message || text;

        // Handle reset requests - don't require a message
        if (reset || userMsg === 'RESET_CONVERSATION') {
            return res.json({ response: 'Conversation reset', emotion: 'neutral', sessionId, message: 'Conversation reset', crisis: false });
        }

        if (!userMsg || userMsg.trim() === '') {
            return res.status(400).json({ error: 'Message is required' });
        }

        let result;
        let usedFastApi = false;

        // 1. Fast path: Forward to the already-running FastAPI server
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 12000); // 15s timeout
            const fastApiResponse = await fetch(`${BACKEND_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: userMsg,
                    advice_given,
                    history
                }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (fastApiResponse.ok) {
                const data = await fastApiResponse.json();
                result = {
                    response: data.response,
                    emotion: data.emotion,
                    confidence: data.confidence,
                    important_words: data.important_words || [],
                    sessionId,
                    crisis: data.crisis === true
                };
                usedFastApi = true;
            } else {
                console.warn(`FastAPI returned HTTP ${fastApiResponse.status}, falling back to Python subprocess.`);
            }
        } catch (fetchErr) {
            console.warn(`Could not reach FastAPI at ${BACKEND_URL}/predict (${fetchErr.message}). Falling back to Python subprocess.`);
        }

        // 2. Fallback path: Old Python subprocess if FastAPI was unreachable
        if (!usedFastApi) {
            result = await callHybridChat(sessionId, userMsg, reset);
            result.crisis = result.crisis === true;
        }

        // Track emotion for book recommendations (only if not a reset)
        if (result.emotion && !reset) {
            updateEmotionFrequency(result.emotion);
            console.log(`Emotion detected: ${result.emotion} -> bucket: ${getEmotionBucket(result.emotion)}`);
            console.log('Current emotion frequency:', emotionFrequency);
        }


        res.json({
            ...result,
            crisis: result.crisis === true
        });
    } catch (error) {
        console.error('Error in /api/chat:', error);
        res.status(500).json({
            error: 'Failed to generate response',
            details: error.message
        });
    }
});

app.post('/predict', async (req, res) => {
    try {
        const { text, message, sessionId = 'default', advice_given = false, history = [] } = req.body;
        const userMsg = text || message || '';

        if (!userMsg || userMsg.trim() === '') {
            return res.status(400).json({ error: 'Text or message is required' });
        }

        let result;
        let usedFastApi = false;

        // Try forwarding to FastAPI
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000);
            const fastApiResponse = await fetch(`${BACKEND_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: userMsg,
                    advice_given,
                    history
                }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (fastApiResponse.ok) {
                const data = await fastApiResponse.json();
                result = {
                    response: data.response,
                    emotion: data.emotion,
                    confidence: data.confidence,
                    important_words: data.important_words || [],
                    sessionId,
                    crisis: data.crisis === true
                };
                usedFastApi = true;
            }
        } catch (e) {
            // Fallback
        }

        if (!usedFastApi) {
            result = await callHybridChat(sessionId, userMsg, false);
            result.crisis = result.crisis === true;
        }

        res.json({
            emotion: result.emotion || 'neutral',
            confidence: result.confidence || 0.95,
            important_words: result.important_words || [],
            response: result.response,
            crisis: result.crisis === true
        });
    } catch (error) {
        console.error('Error in /predict:', error);
        res.status(500).json({
            error: 'Failed to generate response',
            details: error.message
        });
    }
});

app.post('/api/reset', async (req, res) => {
    try {
        const { sessionId = 'default', resetEmotionTracking = false } = req.body;

        // Optionally reset emotion tracking (default: keep it)
        if (resetEmotionTracking) {
            Object.keys(emotionFrequency).forEach(key => {
                emotionFrequency[key] = 0;
            });
            console.log('Emotion tracking reset');
        }

        // Reset is handled by passing reset=true to chat endpoint
        res.json({
            success: true,
            message: 'Chat reset',
            emotionTrackingReset: resetEmotionTracking
        });
    } catch (error) {
        console.error('Error in /api/reset:', error);
        res.status(500).json({ error: 'Failed to reset chat' });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Get most recent emotion for book recommendations
app.get('/api/emotion', (req, res) => {
    const mostRecent = getMostRecentEmotion();
    res.json({
        emotion: mostRecent,
        frequency: emotionFrequency, // Still available for reference
        allEmotions: emotionFrequency
    });
});

// Reset emotion tracking (optional)
app.post('/api/emotion/reset', (req, res) => {
    Object.keys(emotionFrequency).forEach(key => {
        emotionFrequency[key] = 0;
    });
    res.json({ success: true, message: 'Emotion tracking reset' });
});

// Proxy: Book recommendation → FastAPI /recommend-book
app.post('/api/recommend-book', async (req, res) => {
    try {
        const fastApiResponse = await fetch(`${BACKEND_URL}/recommend-book`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });
        const data = await fastApiResponse.json();
        res.status(fastApiResponse.status).json(data);
    } catch (err) {
        console.error('Error proxying /api/recommend-book:', err.message);
        res.status(502).json({ error: 'Book recommendation service unavailable', details: err.message });
    }
});

// Proxy: Occlusion explanation → FastAPI /explain
app.post('/api/explain', async (req, res) => {
    try {
        const fastApiResponse = await fetch(`${BACKEND_URL}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });
        const data = await fastApiResponse.json();
        res.status(fastApiResponse.status).json(data);
    } catch (err) {
        console.error('Error proxying /api/explain:', err.message);
        res.status(502).json({ error: 'Explanation service unavailable', details: err.message });
    }
});

// Serve frontend pages - MUST be before static middleware
const publicPath = path.join(__dirname, 'public');

app.get('/', (req, res) => {
    res.sendFile(path.resolve(publicPath, 'index.html'));
});

app.get('/chat', (req, res) => {
    console.log('GET /chat requested');
    res.sendFile(path.resolve(publicPath, 'chat.html'), (err) => {
        if (err) {
            console.error('Error sending chat.html:', err);
            res.status(404).send('Chat page not found');
        }
    });
});

app.get('/book', (req, res) => {
    console.log('GET /book requested');
    res.sendFile(path.resolve(publicPath, 'book.html'), (err) => {
        if (err) {
            console.error('Error sending book.html:', err);
            res.status(404).send('Book page not found');
        }
    });
});

app.get('/exercise', (req, res) => {
    console.log('GET /exercise requested');
    res.sendFile(path.resolve(publicPath, 'exercise.html'), (err) => {
        if (err) {
            console.error('Error sending exercise.html:', err);
            res.status(404).send('Exercise page not found');
        }
    });
});

app.get('/game', (req, res) => {
    console.log('GET /game requested');
    res.sendFile(path.resolve(publicPath, 'game.html'), (err) => {
        if (err) {
            console.error('Error sending game.html:', err);
            res.status(404).send('Game page not found');
        }
    });
});

// Static files (CSS, JS, images) - serve after specific routes
app.use(express.static(publicPath, {
    index: false // Don't serve index.html for directory requests
}));

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    console.log(`Make sure Ollama is running and qwen2.5:1.5b model is installed`);
    console.log(`Routes available:`);
    console.log(`  - http://localhost:${PORT}/`);
    console.log(`  - http://localhost:${PORT}/chat`);
    console.log(`  - http://localhost:${PORT}/book`);
    console.log(`  - http://localhost:${PORT}/exercise`);
    console.log(`  - http://localhost:${PORT}/game`);
});

