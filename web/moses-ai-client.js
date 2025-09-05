/**
 * Moses Omondi's AI Recruitment Assistant - Website Integration Client
 * ===================================================================
 * 
 * JavaScript client for integrating Moses's AI recruitment assistant 
 * into your website. Supports chat functionality and recruitment conversations.
 * 
 * @version 1.0.0
 * @author Moses Omondi
 */

class MosesAIClient {
    constructor(config = {}) {
        this.apiUrl = config.apiUrl || 'http://localhost:8000';
        this.timeout = config.timeout || 30000; // 30 seconds
        this.retries = config.retries || 3;
        this.debug = config.debug || false;
        
        // Event callbacks
        this.onMessageStart = config.onMessageStart || null;
        this.onMessageComplete = config.onMessageComplete || null;
        this.onError = config.onError || null;
        
        this.log('Moses AI Client initialized', { apiUrl: this.apiUrl });
    }
    
    /**
     * Chat with Moses's AI recruitment assistant
     * @param {string} message - The question or message to send
     * @param {Object} options - Additional options
     * @returns {Promise<Object>} API response with Moses's answer
     */
    async chat(message, options = {}) {
        const requestData = {
            message: message,
            max_tokens: options.maxTokens || 400,
            temperature: options.temperature || 0.7,
            context: options.context || null
        };
        
        this.log('Sending chat request', requestData);
        
        if (this.onMessageStart) {
            this.onMessageStart(message);
        }
        
        try {
            const response = await this.makeRequest('/chat', 'POST', requestData);
            
            if (this.onMessageComplete) {
                this.onMessageComplete(response.response);
            }
            
            return response;
            
        } catch (error) {
            this.log('Chat request failed', error);
            
            if (this.onError) {
                this.onError(error);
            }
            
            throw error;
        }
    }
    
    /**
     * Get Moses's capabilities and expertise areas
     * @returns {Promise<Object>} Capabilities information
     */
    async getCapabilities() {
        return await this.makeRequest('/capabilities', 'GET');
    }
    
    /**
     * Get sample questions for testing
     * @returns {Promise<Object>} Sample questions organized by category
     */
    async getSampleQuestions() {
        return await this.makeRequest('/sample-questions', 'GET');
    }
    
    /**
     * Check API health status
     * @returns {Promise<Object>} Health status information
     */
    async getHealth() {
        return await this.makeRequest('/health', 'GET');
    }
    
    /**
     * Make HTTP request with retry logic
     * @private
     */
    async makeRequest(endpoint, method = 'GET', data = null) {
        const url = `${this.apiUrl}${endpoint}`;
        
        for (let attempt = 1; attempt <= this.retries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);
                
                const options = {
                    method: method,
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    signal: controller.signal
                };
                
                if (data && method !== 'GET') {
                    options.body = JSON.stringify(data);
                }
                
                const response = await fetch(url, options);
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                
                this.log(`Request successful (attempt ${attempt})`, { 
                    url, method, status: response.status 
                });
                
                return result;
                
            } catch (error) {
                this.log(`Request failed (attempt ${attempt}/${this.retries})`, { 
                    url, method, error: error.message 
                });
                
                if (attempt === this.retries) {
                    throw new Error(`Request failed after ${this.retries} attempts: ${error.message}`);
                }
                
                // Wait before retrying (exponential backoff)
                await this.delay(Math.pow(2, attempt - 1) * 1000);
            }
        }
    }
    
    /**
     * Utility function for delays
     * @private
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Logging utility
     * @private
     */
    log(message, data = null) {
        if (this.debug) {
            console.log(`[Moses AI Client] ${message}`, data || '');
        }
    }
}

/**
 * Moses AI Chat Widget
 * Easy-to-use chat widget for website integration
 */
class MosesAIChatWidget {
    constructor(config = {}) {
        this.client = new MosesAIClient(config);
        this.container = config.container || 'moses-ai-chat';
        this.theme = config.theme || 'professional';
        this.welcomeMessage = config.welcomeMessage || 
            "Hi! I'm Moses Omondi's AI recruitment assistant. Ask me about Moses's DevSecOps, MLOps, and AI security expertise!";
        
        this.isInitialized = false;
        this.isTyping = false;
        
        this.init();
    }
    
    /**
     * Initialize the chat widget
     */
    init() {
        if (this.isInitialized) return;
        
        this.createWidget();
        this.attachEventListeners();
        this.addWelcomeMessage();
        
        this.isInitialized = true;
    }
    
    /**
     * Create the chat widget HTML structure
     */
    createWidget() {
        const container = document.getElementById(this.container);
        if (!container) {
            throw new Error(`Container element with ID '${this.container}' not found`);
        }
        
        container.innerHTML = `
            <div class="moses-ai-widget moses-ai-theme-${this.theme}">
                <div class="moses-ai-header">
                    <div class="moses-ai-avatar">
                        <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMjAiIGN5PSIyMCIgcj0iMjAiIGZpbGw9IiMxZTQwYWYiLz4KPGV4dCB4PSIyMCIgeT0iMjYiIGZpbGw9IndoaXRlIiBmb250LXNpemU9IjE0cHgiIGZvbnQtZmFtaWx5PSJBcmlhbCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+TU88L3RleHQ+Cjwvc3ZnPgo=" alt="Moses AI" />
                    </div>
                    <div class="moses-ai-info">
                        <h3>Moses Omondi's AI Assistant</h3>
                        <p>DevSecOps • MLOps • MLSecOps Expert</p>
                    </div>
                </div>
                
                <div class="moses-ai-messages" id="moses-ai-messages">
                    <!-- Messages will be added here -->
                </div>
                
                <div class="moses-ai-typing" id="moses-ai-typing" style="display: none;">
                    <span></span><span></span><span></span>
                    <span class="typing-text">Moses's AI is typing...</span>
                </div>
                
                <div class="moses-ai-input-container">
                    <input 
                        type="text" 
                        id="moses-ai-input" 
                        placeholder="Ask about Moses's DevSecOps, MLOps, or AI security expertise..."
                        maxlength="500"
                    />
                    <button id="moses-ai-send" type="button">
                        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M2 10L18 2L11 10L18 18L2 10Z" stroke="currentColor" stroke-width="1.5"/>
                        </svg>
                    </button>
                </div>
                
                <div class="moses-ai-quick-questions">
                    <button class="quick-question" data-question="Tell me about Moses's DevSecOps experience">DevSecOps Experience</button>
                    <button class="quick-question" data-question="What's Moses's MLOps expertise?">MLOps Expertise</button>
                    <button class="quick-question" data-question="Is Moses qualified for VP-level AI roles?">VP-Level Roles</button>
                </div>
            </div>
        `;
        
        this.injectStyles();
    }
    
    /**
     * Inject CSS styles for the widget
     */
    injectStyles() {
        if (document.getElementById('moses-ai-styles')) return;
        
        const styles = `
            <style id="moses-ai-styles">
                .moses-ai-widget {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e5e9;
                    border-radius: 12px;
                    background: white;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                    max-width: 400px;
                    height: 500px;
                    display: flex;
                    flex-direction: column;
                }
                
                .moses-ai-header {
                    padding: 16px;
                    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                    color: white;
                    display: flex;
                    align-items: center;
                    border-radius: 12px 12px 0 0;
                }
                
                .moses-ai-avatar img {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    margin-right: 12px;
                }
                
                .moses-ai-info h3 {
                    margin: 0 0 4px 0;
                    font-size: 16px;
                    font-weight: 600;
                }
                
                .moses-ai-info p {
                    margin: 0;
                    font-size: 12px;
                    opacity: 0.9;
                }
                
                .moses-ai-messages {
                    flex: 1;
                    padding: 16px;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                
                .message {
                    max-width: 80%;
                    padding: 12px 16px;
                    border-radius: 18px;
                    word-wrap: break-word;
                }
                
                .message.user {
                    background: #1e40af;
                    color: white;
                    align-self: flex-end;
                    margin-left: auto;
                }
                
                .message.assistant {
                    background: #f1f5f9;
                    color: #334155;
                    align-self: flex-start;
                    border: 1px solid #e2e8f0;
                }
                
                .moses-ai-typing {
                    padding: 12px 16px;
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    color: #64748b;
                    font-size: 14px;
                }
                
                .moses-ai-typing span:not(.typing-text) {
                    width: 8px;
                    height: 8px;
                    background: #94a3b8;
                    border-radius: 50%;
                    animation: typing 1.4s infinite ease-in-out;
                }
                
                .moses-ai-typing span:nth-child(1) { animation-delay: -0.32s; }
                .moses-ai-typing span:nth-child(2) { animation-delay: -0.16s; }
                
                @keyframes typing {
                    0%, 80%, 100% { transform: scale(0); }
                    40% { transform: scale(1); }
                }
                
                .moses-ai-input-container {
                    padding: 16px;
                    border-top: 1px solid #e2e8f0;
                    display: flex;
                    gap: 8px;
                }
                
                #moses-ai-input {
                    flex: 1;
                    padding: 12px 16px;
                    border: 1px solid #d1d5db;
                    border-radius: 24px;
                    outline: none;
                    font-size: 14px;
                }
                
                #moses-ai-input:focus {
                    border-color: #1e40af;
                    box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1);
                }
                
                #moses-ai-send {
                    padding: 12px;
                    background: #1e40af;
                    color: white;
                    border: none;
                    border-radius: 50%;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                
                #moses-ai-send:hover {
                    background: #1d4ed8;
                }
                
                #moses-ai-send:disabled {
                    background: #9ca3af;
                    cursor: not-allowed;
                }
                
                .moses-ai-quick-questions {
                    padding: 8px 16px 16px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                }
                
                .quick-question {
                    padding: 6px 12px;
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 16px;
                    font-size: 12px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .quick-question:hover {
                    background: #1e40af;
                    color: white;
                    border-color: #1e40af;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    /**
     * Attach event listeners
     */
    attachEventListeners() {
        const input = document.getElementById('moses-ai-input');
        const sendButton = document.getElementById('moses-ai-send');
        const quickQuestions = document.querySelectorAll('.quick-question');
        
        sendButton.addEventListener('click', () => this.sendMessage());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        quickQuestions.forEach(button => {
            button.addEventListener('click', () => {
                const question = button.dataset.question;
                this.sendMessage(question);
            });
        });
    }
    
    /**
     * Add welcome message
     */
    addWelcomeMessage() {
        this.addMessage(this.welcomeMessage, 'assistant');
    }
    
    /**
     * Send a message to Moses's AI
     */
    async sendMessage(message = null) {
        const input = document.getElementById('moses-ai-input');
        const sendButton = document.getElementById('moses-ai-send');
        
        if (!message) {
            message = input.value.trim();
        }
        
        if (!message || this.isTyping) return;
        
        // Add user message
        this.addMessage(message, 'user');
        input.value = '';
        
        // Show typing indicator
        this.showTyping(true);
        sendButton.disabled = true;
        
        try {
            const response = await this.client.chat(message);
            this.addMessage(response.response, 'assistant');
            
        } catch (error) {
            this.addMessage(
                "I apologize, but I'm having trouble connecting right now. Please try again in a moment.",
                'assistant'
            );
        } finally {
            this.showTyping(false);
            sendButton.disabled = false;
        }
    }
    
    /**
     * Add a message to the chat
     */
    addMessage(text, sender) {
        const messagesContainer = document.getElementById('moses-ai-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    /**
     * Show/hide typing indicator
     */
    showTyping(show) {
        const typingElement = document.getElementById('moses-ai-typing');
        typingElement.style.display = show ? 'flex' : 'none';
        this.isTyping = show;
        
        if (show) {
            const messagesContainer = document.getElementById('moses-ai-messages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MosesAIClient, MosesAIChatWidget };
}

// Make available globally
window.MosesAIClient = MosesAIClient;
window.MosesAIChatWidget = MosesAIChatWidget;
