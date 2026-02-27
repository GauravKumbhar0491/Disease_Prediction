/* =========================================
   MediAI — Premium Chat Script
   ========================================= */

// ── Init ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    spawnParticles();
    setupTextarea();
    setupSendButton();
    setupKeyboardShortcuts();
    setupSidebarToggle();
    setupClearButton();
    setupNavItems();
});

// ── Particle effect ───────────────────────────────
function spawnParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    for (let i = 0; i < 28; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const size = Math.random() * 4 + 2;
        p.style.cssText = `
            width:${size}px; height:${size}px;
            left:${Math.random() * 100}%;
            bottom:${Math.random() * 40}%;
            --dur:${(Math.random() * 8 + 5).toFixed(1)}s;
            --delay:-${(Math.random() * 10).toFixed(1)}s;
            opacity:0;
        `;
        container.appendChild(p);
    }
}

// ── Textarea auto-resize ──────────────────────────
function setupTextarea() {
    const ta = document.getElementById('userInput');
    if (!ta) return;
    ta.addEventListener('input', () => {
        ta.style.height = 'auto';
        ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
    });
}

// ── Send button ───────────────────────────────────
function setupSendButton() {
    const btn = document.getElementById('sendBtn');
    if (btn) btn.addEventListener('click', handleSend);
}

// ── Keyboard shortcuts ────────────────────────────
function setupKeyboardShortcuts() {
    const ta = document.getElementById('userInput');
    if (!ta) return;
    ta.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
}

// ── Sidebar toggle (mobile) ───────────────────────
function setupSidebarToggle() {
    const toggleBtn = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    if (!toggleBtn || !sidebar) return;

    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (sidebar.classList.contains('open') &&
            !sidebar.contains(e.target) &&
            !toggleBtn.contains(e.target)) {
            sidebar.classList.remove('open');
        }
    });
}

// ── Clear conversation ────────────────────────────
function setupClearButton() {
    const clearBtn = document.getElementById('clearBtn');
    const messages = document.getElementById('messages');
    if (!clearBtn || !messages) return;

    clearBtn.addEventListener('click', () => {
        // Animate out existing messages
        const existing = messages.querySelectorAll('.message');
        existing.forEach(m => {
            m.style.opacity = '0';
            m.style.transition = 'opacity 0.2s ease';
        });
        setTimeout(() => {
            // Keep only welcome block
            const toRemove = messages.querySelectorAll('.message');
            toRemove.forEach(m => m.remove());
        }, 250);
    });
}

// ── Nav item selection ────────────────────────────
function setupNavItems() {
    const items = document.querySelectorAll('.nav-item');
    items.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            items.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

// ── Main send handler ────────────────────────────
function handleSend() {
    const ta = document.getElementById('userInput');
    if (!ta) return;
    const userMessage = ta.value.trim();
    if (!userMessage) return;

    // Hide welcome block on first message
    const welcome = document.querySelector('.welcome-block');
    if (welcome) {
        welcome.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        welcome.style.opacity = '0';
        welcome.style.transform = 'translateY(-10px)';
        setTimeout(() => welcome.remove(), 300);
    }

    addMessage(userMessage, 'user-message');
    ta.value = '';
    ta.style.height = 'auto';

    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) sendBtn.disabled = true;

    const typingId = showTypingIndicator();
    sendMessageToBackend(userMessage, typingId);
}

// ── Quick chip handler (global) ────────────────────
window.quickAsk = function (text) {
    const ta = document.getElementById('userInput');
    if (!ta) return;
    ta.value = text;
    ta.dispatchEvent(new Event('input'));
    ta.focus();
    handleSend();
};

// ── Add message bubble ────────────────────────────
function addMessage(content, type) {
    const messages = document.getElementById('messages');
    if (!messages) return;

    const isUser = type === 'user-message';

    const wrapper = document.createElement('div');
    wrapper.classList.add('message', type);

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = isUser ? 'You' : 'AI';

    // Content block
    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';

    const sender = document.createElement('div');
    sender.className = 'msg-sender';
    sender.textContent = isUser ? 'You' : 'MediAI';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = marked.parse(content);

    contentDiv.appendChild(sender);
    contentDiv.appendChild(bubble);

    if (isUser) {
        wrapper.appendChild(contentDiv);
        wrapper.appendChild(avatar);
    } else {
        wrapper.appendChild(avatar);
        wrapper.appendChild(contentDiv);
    }

    messages.appendChild(wrapper);
    scrollToBottom();
}

// ── Typing indicator ──────────────────────────────
function showTypingIndicator() {
    const messages = document.getElementById('messages');
    if (!messages) return null;

    const id = 'typing-' + Date.now();
    const wrapper = document.createElement('div');
    wrapper.className = 'typing-indicator';
    wrapper.id = id;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'typing-bubble';

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        bubble.appendChild(dot);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    messages.appendChild(wrapper);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── Backend fetch ─────────────────────────────────
function sendMessageToBackend(userMessage, typingId) {
    fetch('/api/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
    })
        .then(response => response.json())
        .then(data => {
            removeTypingIndicator(typingId);
            const aiMessage = data.response || 'I could not process that. Please try again.';
            addMessage(aiMessage, 'ai-message');
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator(typingId);
            addMessage('Sorry, there was an error connecting to the server. Please try again.', 'error-message ai-message');
        })
        .finally(() => {
            const sendBtn = document.getElementById('sendBtn');
            if (sendBtn) sendBtn.disabled = false;
            const ta = document.getElementById('userInput');
            if (ta) ta.focus();
        });
}

// ── Scroll to bottom ──────────────────────────────
function scrollToBottom() {
    const wrapper = document.querySelector('.messages-wrapper');
    if (wrapper) wrapper.scrollTop = wrapper.scrollHeight;
}
