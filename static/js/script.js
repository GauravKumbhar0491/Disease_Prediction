document.getElementById('sendBtn').addEventListener('click', function() {
    const inputField = document.getElementById('userInput');
    const userMessage = inputField.value.trim();

    if (userMessage) {
        addMessage(userMessage, 'user-message');
        inputField.value = '';
        // Instead of simulating, we will send the message to the Flask backend
        sendMessageToBackend(userMessage);
    }
});

function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type);
    messageDiv.innerHTML = marked.parse(content);
    document.getElementById('messages').appendChild(messageDiv);
    scrollToBottom();
}

function sendMessageToBackend(userMessage) {
    fetch('/api/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
    })
        .then(response => response.json())
        .then(data => {
            // Add the response from the backend (AI's response) to the chat
            const aiMessage = data.response;  // The response from your Flask API
            addMessage(aiMessage, 'ai-message');
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Sorry, there was an error processing your message.', 'error-message');
        });
}

function scrollToBottom() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
