const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

app.post('/api/chat', (req, res) => {
  const userMessage = req.body.message;
  console.log('User:', userMessage);

  let reply = "I got your message: " + userMessage;

  if (userMessage.toLowerCase().includes("admission")) {
    reply = "Our admission process starts in March and ends in July. Please visit the admission section on our website.";
  } else if (userMessage.toLowerCase().includes("fee")) {
    reply = "The average annual fee is around ₹80,000, depending on the course.";
  }

  res.json({ reply });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
