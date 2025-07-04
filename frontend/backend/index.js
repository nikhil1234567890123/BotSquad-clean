const express = require('express');
const cors = require('cors');
const OpenAI = require("openai");

const app = express();
const openai = new OpenAI();
const PORT = 5000;

app.use(cors());
app.use(express.json());

app.post('/api/chat', async (req, res) => {
    const userMessage = req.body.message;

    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                { role: "user", content: userMessage }
            ],
        });

        res.json({ reply: completion.choices[0].message.content });
    } catch (error) {
        console.error(error);
        res.status(500).json({ reply: "Sorry, something went wrong." });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
