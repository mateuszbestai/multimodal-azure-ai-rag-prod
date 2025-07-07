import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Copy, Download, Image as ImageIcon, Search, X, ChevronRight, ChevronLeft, Maximize2, ZoomIn, ZoomOut, Plus, Trash2, MessageSquare, Edit2, Save, MoreHorizontal, ArrowLeft, Moon, Sun } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { formatDistanceToNow } from 'date-fns';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { CSSTransition, TransitionGroup } from 'react-transition-group';

// API Configuration
const API_CONFIG = {
  baseUrl: import.meta.env.VITE_API_URL || 'http://localhost:5001',
  endpoints: {
    chat: '/api/chat',
    chatStream: '/api/chat/stream',
    health: '/api/health',
    imageProxy: '/api/image/proxy'
  }
};

// Helper function to build full image URLs
const buildImageUrl = (url: string | undefined): string | undefined => {
  if (!url) return undefined;
  
  // If it's already a full URL, return as-is
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }
  
  // If it's a relative URL (our proxy), prepend the API base URL
  if (url.startsWith('/api/image/proxy')) {
    return `${API_CONFIG.baseUrl}${url}`;
  }
  
  return url;
};

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  sources?: {
    pages: number[];
    images?: string[];
  };
  sourcePreviews?: SourcePreview[];
  loading?: boolean;
  streaming?: boolean;
}

interface SourcePreview {
  id: string;
  page: number;
  content: string;
  imageUrl?: string;
  category?: string;
  title?: string;
  date?: Date;
}

interface ChatSession {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  messages: Message[];
}

interface ImageViewerProps {
  url: string;
  onClose: () => void;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ url, onClose }) => {
  const [scale, setScale] = useState(1);

  return (
    <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center animate-fade-in">
      <div className="absolute top-4 right-4 flex gap-2">
        <button
          onClick={() => setScale(s => s + 0.25)}
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
        >
          <ZoomIn size={20} className="text-white" />
        </button>
        <button
          onClick={() => setScale(s => Math.max(0.5, s - 0.25))}
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
        >
          <ZoomOut size={20} className="text-white" />
        </button>
        <button
          onClick={onClose}
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
        >
          <X size={20} className="text-white" />
        </button>
      </div>
      <img
        src={url}
        alt="Full size preview"
        className="max-w-[90vw] max-h-[90vh] object-contain transition-transform duration-200"
        style={{ transform: `scale(${scale})` }}
      />
    </div>
  );
};

// Health check function
async function checkAPIHealth() {
  try {
    const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.health}`);
    if (!response.ok) {
      console.error('API health check failed:', response.status);
      return false;
    }
    const data = await response.json();
    console.log('API health check passed:', data);
    return true;
  } catch (error) {
    console.error('API health check error:', error);
    return false;
  }
}

function App() {
  // Dark mode state
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });
  
  // Chat history state
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [showChatSidebar, setShowChatSidebar] = useState(true);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  
  // Current chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showReferences, setShowReferences] = useState(false);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Check API health on mount
  useEffect(() => {
    checkAPIHealth().then(isHealthy => {
      if (!isHealthy) {
        console.error(`API not reachable at ${API_CONFIG.baseUrl}. Make sure the backend is running on port 5001.`);
      }
    });
  }, []);

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  // Load chat sessions from localStorage on initial render
  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions');
    if (savedSessions) {
      try {
        const parsedSessions = JSON.parse(savedSessions, (key, value) => {
          // Convert string dates back to Date objects
          if (key === 'timestamp' || key === 'createdAt' || key === 'updatedAt') {
            return new Date(value);
          }
          return value;
        });
        setChatSessions(parsedSessions);
        
        // Set the most recent chat as active if available
        if (parsedSessions.length > 0) {
          const mostRecentChat = parsedSessions.sort((a: ChatSession, b: ChatSession) => 
            new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
          )[0];
          setActiveChatId(mostRecentChat.id);
          setMessages(mostRecentChat.messages);
        } else {
          createNewChat();
        }
      } catch (error) {
        console.error('Error parsing saved chat sessions:', error);
        createNewChat();
      }
    } else {
      createNewChat();
    }
  }, []);

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    }
  }, [chatSessions]);

  // Set current chat data when active chat changes
  useEffect(() => {
    if (activeChatId) {
      const activeChat = chatSessions.find(chat => chat.id === activeChatId);
      if (activeChat) {
        setMessages(activeChat.messages);
      }
    }
  }, [activeChatId, chatSessions]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const createNewChat = () => {
    const newChatId = crypto.randomUUID();
    const newChat: ChatSession = {
      id: newChatId,
      title: `New Chat ${new Date().toLocaleDateString()}`,
      createdAt: new Date(),
      updatedAt: new Date(),
      messages: []
    };
    
    setChatSessions(prev => [newChat, ...prev]);
    setActiveChatId(newChatId);
    setMessages([]);
    setInput('');
  };

  const deleteChat = (chatId: string) => {
    setChatSessions(prev => prev.filter(chat => chat.id !== chatId));
    
    if (chatId === activeChatId) {
      const remainingChats = chatSessions.filter(chat => chat.id !== chatId);
      if (remainingChats.length > 0) {
        setActiveChatId(remainingChats[0].id);
        setMessages(remainingChats[0].messages);
      } else {
        createNewChat();
      }
    }
  };

  const startTitleEdit = (chatId: string) => {
    const chat = chatSessions.find(c => c.id === chatId);
    if (chat) {
      setEditTitle(chat.title);
      setIsEditingTitle(true);
    }
  };

  const saveTitle = () => {
    if (activeChatId && editTitle.trim()) {
      setChatSessions(prev => 
        prev.map(chat => 
          chat.id === activeChatId 
            ? { ...chat, title: editTitle.trim() } 
            : chat
        )
      );
      setIsEditingTitle(false);
    }
  };

  const updateChatSession = (newMessages: Message[]) => {
    if (activeChatId) {
      setChatSessions(prev => 
        prev.map(chat => 
          chat.id === activeChatId 
            ? { 
                ...chat, 
                messages: newMessages, 
                updatedAt: new Date()
              } 
            : chat
        )
      );
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !activeChatId) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      content: input,
      role: 'user',
      timestamp: new Date(),
    };

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      loading: true,
      streaming: true,
      sourcePreviews: [],
    };

    const updatedMessages = [...messages, userMessage, assistantMessage];
    setMessages(updatedMessages);
    updateChatSession(updatedMessages);
    setInput('');
    setIsLoading(true);

    try {
      // Cancel any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new AbortController for this request
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Use fetch API with streaming support
      const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.chatStream}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) {
        throw new Error('No response body');
      }

      let accumulatedContent = '';
      let metadata: any = null;
      let currentSourcePreviews: SourcePreview[] = [];
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              
              if (data.type === 'metadata') {
                metadata = data.data;
                
                // Update source previews for this specific message
                if (metadata.sourcePreviews) {
                  currentSourcePreviews = metadata.sourcePreviews.map((preview: any) => ({
                    ...preview,
                    id: crypto.randomUUID(),
                  }));
                  
                  // Update the assistant message with source previews
                  const streamingMessages = updatedMessages.map(msg =>
                    msg.id === assistantMessage.id
                      ? {
                          ...msg,
                          content: accumulatedContent,
                          loading: false,
                          streaming: true,
                          sourcePreviews: currentSourcePreviews,
                        }
                      : msg
                  );
                  setMessages(streamingMessages);
                  
                  // If this is the current message, show references
                  setSelectedMessageId(assistantMessage.id);
                  setShowReferences(true);
                }
              } else if (data.type === 'chunk') {
                accumulatedContent += data.data;
                
                // Update the assistant message with accumulated content
                const streamingMessages = updatedMessages.map(msg =>
                  msg.id === assistantMessage.id
                    ? {
                        ...msg,
                        content: accumulatedContent,
                        loading: false,
                        streaming: true,
                        sourcePreviews: currentSourcePreviews,
                      }
                    : msg
                );
                setMessages(streamingMessages);
              } else if (data.type === 'done') {
                // Finalize the message
                const finalMessages = updatedMessages.map(msg =>
                  msg.id === assistantMessage.id
                    ? {
                        ...msg,
                        content: accumulatedContent || 'No response could be generated',
                        loading: false,
                        streaming: false,
                        sources: metadata ? {
                          pages: metadata.pages || [],
                          images: metadata.images || [],
                        } : undefined,
                        sourcePreviews: currentSourcePreviews,
                      }
                    : msg
                );
                
                setMessages(finalMessages);
                setIsLoading(false);
                
                // Update chat title based on first user message if it's a new chat
                const activeChat = chatSessions.find(chat => chat.id === activeChatId);
                if (activeChat && activeChat.messages.length === 0) {
                  const newTitle = userMessage.content.length > 30 
                    ? userMessage.content.substring(0, 30) + '...' 
                    : userMessage.content;
                  
                  setChatSessions(prev => 
                    prev.map(chat => 
                      chat.id === activeChatId 
                        ? { ...chat, title: newTitle } 
                        : chat
                    )
                  );
                }
                
                updateChatSession(finalMessages);
              } else if (data.type === 'error') {
                throw new Error(data.message);
              }
            } catch (parseError) {
              console.error('Error parsing JSON:', parseError, 'Line:', line);
            }
          }
        }
      }

      // Process any remaining data in buffer
      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer);
          // Process the final data chunk if needed
        } catch (e) {
          console.error('Error parsing final buffer:', e);
        }
      }

    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('Request was aborted');
        return;
      }

      console.error('API request failed:', error);
      console.error(`Attempted URL: ${API_CONFIG.baseUrl}${API_CONFIG.endpoints.chatStream}`);

      // Try fallback to non-streaming endpoint
      try {
        const fallbackResponse = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.chat}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: input }),
        });

        if (!fallbackResponse.ok) throw new Error(`HTTP error! status: ${fallbackResponse.status}`);

        const data = await fallbackResponse.json();

        let fallbackSourcePreviews: SourcePreview[] = [];
        if (data.sourcePreviews) {
          fallbackSourcePreviews = data.sourcePreviews.map((preview: any) => ({
            ...preview,
            id: crypto.randomUUID(),
          }));
        }

        const finalMessages = updatedMessages.map(msg =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: data.response || 'No response could be generated',
                loading: false,
                streaming: false,
                sources: {
                  pages: data.sources?.pages || [],
                  images: data.sources?.images || [],
                },
                sourcePreviews: fallbackSourcePreviews,
              }
            : msg
        );

        setMessages(finalMessages);
        updateChatSession(finalMessages);
        
        if (fallbackSourcePreviews.length > 0) {
          setSelectedMessageId(assistantMessage.id);
          setShowReferences(true);
        }
      } catch (fallbackError) {
        console.error('Fallback API request failed:', fallbackError);
        
        const errorMessages = updatedMessages.map(msg =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: `Error: Unable to connect to the API at ${API_CONFIG.baseUrl}. Please ensure the backend server is running on port 5001.`,
                loading: false,
                streaming: false,
              }
            : msg
        );
        
        setMessages(errorMessages);
        updateChatSession(errorMessages);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const exportChat = () => {
    if (!activeChatId) return;
    
    const activeChat = chatSessions.find(chat => chat.id === activeChatId);
    if (!activeChat) return;
    
    const chatExport = activeChat.messages
      .map(msg => `${msg.role.toUpperCase()} (${msg.timestamp.toISOString()}):\n${msg.content}\n`)
      .join('\n---\n\n');
    
    const blob = new Blob([chatExport], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${activeChat.title}-${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearChat = () => {
    if (activeChatId) {
      setMessages([]);
      updateChatSession([]);
    }
  };

  const filteredSourcePreviews = (() => {
    const selectedMessage = messages.find(msg => msg.id === selectedMessageId);
    const sourcePreviews = selectedMessage?.sourcePreviews || [];
    
    return sourcePreviews.filter(preview => {
      const matchesSearch = searchTerm
        ? preview.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
          preview.title?.toLowerCase().includes(searchTerm.toLowerCase())
        : true;
      const matchesCategory = categoryFilter
        ? preview.category === categoryFilter
        : true;
      return matchesSearch && matchesCategory;
    });
  })();

  const categories = (() => {
    const selectedMessage = messages.find(msg => msg.id === selectedMessageId);
    const sourcePreviews = selectedMessage?.sourcePreviews || [];
    return Array.from(
      new Set(sourcePreviews.map(preview => preview.category).filter(Boolean))
    );
  })();

  // Current active chat (if any)
  const activeChat = activeChatId 
    ? chatSessions.find(chat => chat.id === activeChatId) 
    : null;

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      {/* Chat History Sidebar */}
      <div 
        className={`w-72 bg-gray-900 dark:bg-gray-950 transform transition-transform duration-300 ease-in-out ${
          showChatSidebar ? 'translate-x-0' : '-translate-x-full'
        } absolute md:relative z-20 h-full overflow-hidden shadow-lg md:shadow-none`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 border-b border-gray-800 dark:border-gray-700">
            <button 
              onClick={createNewChat}
              className="w-full flex items-center justify-center gap-2 bg-primary-600 hover:bg-primary-700 text-white py-2 px-4 rounded-lg transition-colors"
            >
              <Plus size={18} />
              New Chat
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            <div className="p-2">
              <h2 className="text-gray-400 dark:text-gray-500 text-xs uppercase font-semibold mb-2 px-2">Chat History</h2>
              <div className="space-y-1">
                {chatSessions.map(chat => (
                  <div 
                    key={chat.id}
                    className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-colors ${
                      chat.id === activeChatId 
                        ? 'bg-gray-800 dark:bg-gray-700 text-white' 
                        : 'text-gray-300 hover:bg-gray-800/70 dark:hover:bg-gray-700/70'
                    }`}
                    onClick={() => setActiveChatId(chat.id)}
                  >
                    <div className="flex items-center gap-2 overflow-hidden">
                      <MessageSquare size={16} />
                      <span className="text-sm truncate">{chat.title}</span>
                    </div>
                    <div className={`opacity-0 group-hover:opacity-100 transition-opacity`}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          startTitleEdit(chat.id);
                        }}
                        className="p-1 text-gray-400 hover:text-white transition-colors"
                      >
                        <Edit2 size={14} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteChat(chat.id);
                        }}
                        className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="p-4 border-t border-gray-800 dark:border-gray-700 text-gray-400 dark:text-gray-500 text-xs">
            <p>Your chats are stored locally in this browser</p>
          </div>
        </div>
      </div>

      {/* Chat Sidebar Toggle */}
      <button
        onClick={() => setShowChatSidebar(!showChatSidebar)}
        className="fixed left-0 top-6 z-30 md:hidden bg-primary-600 text-white rounded-r-lg p-2"
      >
        {showChatSidebar ? <ArrowLeft size={20} /> : <MessageSquare size={20} />}
      </button>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 shadow-sm transition-colors duration-200">
          <div className="max-w-4xl mx-auto flex justify-between items-center">
            {isEditingTitle && activeChat ? (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  className="text-lg border-b border-gray-300 dark:border-gray-600 bg-transparent text-gray-800 dark:text-gray-200 focus:outline-none focus:border-primary-500 font-medium"
                  autoFocus
                />
                <button 
                  onClick={saveTitle}
                  className="p-1 text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400"
                >
                  <Save size={18} />
                </button>
              </div>
            ) : (
              <h1 className="text-xl font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                {activeChat ? (
                  <>
                    <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                      <span className="text-white font-bold">AI</span>
                    </div>
                    <span className="truncate max-w-sm">
                      {activeChat.title}
                    </span>
                    <button 
                      onClick={() => startTitleEdit(activeChat.id)}
                      className="p-1 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300"
                    >
                      <Edit2 size={16} />
                    </button>
                  </>
                ) : (
                  <>
                    <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                      <span className="text-white font-bold">AI</span>
                    </div>
                    Knowledge Assistant
                  </>
                )}
              </h1>
            )}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
                title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
              >
                {darkMode ? <Sun size={16} /> : <Moon size={16} />}
              </button>
              <button
                onClick={exportChat}
                className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
              >
                <Download size={16} />
                Export
              </button>
              <button
                onClick={clearChat}
                className="flex items-center gap-2 px-3 py-1.5 text-sm text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors"
              >
                <X size={16} />
                Clear
              </button>
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="w-16 h-16 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                    <MessageSquare size={28} className="text-primary-600 dark:text-primary-400" />
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-2">Start a new conversation</h2>
                  <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md">
                    Ask any question about the documents in your knowledge base.
                  </p>
                </div>
              </div>
            ) : (
              <TransitionGroup className="space-y-6">
                {messages.map((message) => (
                  <CSSTransition key={message.id} timeout={300} classNames="message">
                    <div
                      className={`flex ${
                        message.role === 'user' ? 'justify-end' : 'justify-start'
                      } animate-slide-in`}
                    >
                      {message.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-full bg-primary-600 flex items-center justify-center mr-3 mt-1">
                          <span className="text-white text-sm font-semibold">AI</span>
                        </div>
                      )}
                      <div
                        className={`max-w-[80%] rounded-lg p-4 ${
                          message.role === 'user'
                            ? 'bg-primary-600 text-white'
                            : 'bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700'
                        } transition-colors duration-200`}
                      >
                        {message.loading ? (
                          <div className="flex items-center gap-2">
                            <Loader2 className="animate-spin" size={16} />
                            <span>Processing your request...</span>
                          </div>
                        ) : (
                          <div className="space-y-2">
                            <div className={`prose prose-sm max-w-none ${
                              message.role === 'user' ? 'prose-invert' : 'dark:prose-invert'
                            }`}>
                              <ReactMarkdown
                                components={{
                                  code({ node, inline, className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || '');
                                    return !inline && match ? (
                                      <SyntaxHighlighter
                                        style={darkMode ? oneDark : tomorrow}
                                        language={match[1]}
                                        PreTag="div"
                                        {...props}
                                      >
                                        {String(children).replace(/\n$/, '')}
                                      </SyntaxHighlighter>
                                    ) : (
                                      <code className={className} {...props}>
                                        {children}
                                      </code>
                                    );
                                  },
                                }}
                              >
                                {message.content}
                              </ReactMarkdown>
                            </div>
                            {message.streaming && (
                              <span className="inline-block w-1 h-4 bg-gray-400 dark:bg-gray-500 animate-pulse ml-1"></span>
                            )}
                            <div className="flex items-center justify-between mt-2 text-sm">
                              <span className={`${
                                message.role === 'user'
                                  ? 'text-primary-100'
                                  : 'text-gray-400 dark:text-gray-500'
                              }`}>
                                {formatDistanceToNow(message.timestamp, { addSuffix: true })}
                              </span>
                              <div className="flex items-center gap-2">
                                {message.role === 'assistant' && message.sourcePreviews && message.sourcePreviews.length > 0 && (
                                  <button
                                    onClick={() => {
                                      setSelectedMessageId(message.id);
                                      setShowReferences(true);
                                    }}
                                    className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
                                      selectedMessageId === message.id && showReferences
                                        ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                                        : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                    } transition-colors`}
                                    title="View source references"
                                  >
                                    <Search size={14} />
                                    <span>{message.sourcePreviews.length} source{message.sourcePreviews.length > 1 ? 's' : ''}</span>
                                  </button>
                                )}
                                <button
                                  onClick={() => copyToClipboard(message.content)}
                                  className={`p-1 ${
                                    message.role === 'user'
                                      ? 'text-primary-100 hover:text-white'
                                      : 'text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300'
                                  } transition-colors`}
                                  title="Copy message"
                                >
                                  <Copy size={16} />
                                </button>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                      {message.role === 'user' && (
                        <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center ml-3 mt-1">
                          <span className="text-gray-600 dark:text-gray-300 text-sm font-semibold">U</span>
                        </div>
                      )}
                    </div>
                  </CSSTransition>
                ))}
              </TransitionGroup>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Form */}
        <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-4 transition-colors duration-200">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="flex gap-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question..."
                className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 transition-colors duration-200"
                disabled={isLoading}
              />
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                >
                  {isLoading ? (
                    <Loader2 className="animate-spin" size={16} />
                  ) : (
                    <Send size={16} />
                  )}
                  Send
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* References Sidebar */}
      <div
        className={`w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 transform transition-all duration-300 ease-in-out ${
          showReferences ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200">References</h2>
                {selectedMessageId && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {filteredSourcePreviews.length} source{filteredSourcePreviews.length !== 1 ? 's' : ''} found
                  </p>
                )}
              </div>
              <button
                onClick={() => {
                  setShowReferences(false);
                  setSelectedMessageId(null);
                }}
                className="p-1 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="relative">
              <input
                type="text"
                placeholder="Search references..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-2 pr-10 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors duration-200"
              />
              <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" size={18} />
            </div>
            {categories.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {categories.map((category) => (
                  <button
                    key={category}
                    onClick={() => setCategoryFilter(categoryFilter === category ? null : category || null)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      categoryFilter === category
                        ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-800 dark:text-primary-200'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    } transition-colors`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <div className="space-y-4">
              {filteredSourcePreviews.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-gray-500 dark:text-gray-400">
                    {selectedMessageId 
                      ? searchTerm || categoryFilter 
                        ? 'No matching references found' 
                        : 'No source references for this message'
                      : 'Select a message to view its references'}
                  </p>
                </div>
              ) : (
                filteredSourcePreviews.map((preview) => (
                  <div
                    key={preview.id}
                    className="bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 p-4 hover:border-primary-300 dark:hover:border-primary-600 transition-all duration-200"
                  >
                    {preview.title && (
                      <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-1">{preview.title}</h3>
                    )}
                    <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-2">
                      <span>Page {preview.page}</span>
                      {preview.date && (
                        <>
                          <span>•</span>
                          <span>{formatDistanceToNow(preview.date, { addSuffix: true })}</span>
                        </>
                      )}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">{preview.content}</p>
                    {preview.imageUrl && (
                      <div className="relative aspect-video bg-gray-50 dark:bg-gray-800 rounded-md overflow-hidden group">
                        <img
                          src={buildImageUrl(preview.imageUrl)}
                          alt={preview.title || `Preview from page ${preview.page}`}
                          className="w-full h-full object-contain p-2"
                          loading="lazy"
                          onError={(e) => {
                            console.error('Failed to load image:', preview.imageUrl);
                            e.currentTarget.style.display = 'none';
                          }}
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-3">
                          <button
                            onClick={() => preview.imageUrl && setSelectedImage(preview.imageUrl)}
                            className="text-white text-sm hover:text-primary-200 flex items-center gap-1"
                          >
                            <Maximize2 size={14} />
                            <span>View Full Size</span>
                          </button>
                          <a
                            href={buildImageUrl(preview.imageUrl)}
                            download
                            className="text-white text-sm hover:text-primary-200 flex items-center gap-1"
                          >
                            <Download size={14} />
                            <span>Download</span>
                          </a>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* References Toggle */}
      <button
        onClick={() => setShowReferences(!showReferences)}
        className={`fixed right-0 top-1/2 transform -translate-y-1/2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-l-lg p-2 shadow-md transition-all duration-200 ${
          showReferences ? 'translate-x-96' : ''
        }`}
      >
        {showReferences ? (
          <ChevronRight className="text-gray-600 dark:text-gray-400" size={20} />
        ) : (
          <ChevronLeft className="text-gray-600 dark:text-gray-400" size={20} />
        )}
      </button>

      {/* Image Viewer */}
      {selectedImage && (
        <ImageViewer url={buildImageUrl(selectedImage) || ''} onClose={() => setSelectedImage(null)} />
      )}
    </div>
  );
}

export default App;