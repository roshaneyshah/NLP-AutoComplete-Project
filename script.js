/* ==========================================================================
   SCRIBE NLP - CORE LOGIC & INTERACTIVE CLIENT ENGINE
   Contains:
   - High-fidelity client-side N-Gram Language Model
   - Dynamic UI Auto-complete & Tab completions
   - Editorial Transitions, Tab Managers & Intersection Observers
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    
    // ==========================================================================
    // 1. DATA CORPUS & NLP MODEL ENGINE
    // ==========================================================================
    
    // Sample corpus representing standard tweets & synthetic validation sentences
    const corpus = [
        "i love this song",
        "how are you today",
        "the weather is nice",
        "good morning everyone",
        "i am feeling great",
        "python is awesome",
        "machine learning rocks",
        "artificial intelligence is cool",
        "i hate monday mornings",
        "you are amazing",
        "i love this python programming",
        "how are you feeling today",
        "the weather is nice and great today",
        "good morning everyone how are you today",
        "i am feeling great and amazing",
        "python is awesome for machine learning",
        "machine learning rocks the world of tech",
        "artificial intelligence is cool and powerful",
        "i hate monday mornings and traffic",
        "you are amazing and brilliant",
        "i love this nice weather",
        "how are you today my friend",
        "the weather is nice today",
        "good morning everyone and welcome",
        "i am feeling great today",
        "python is awesome to learn",
        "machine learning rocks and makes smart models",
        "artificial intelligence is cool for automation",
        "i hate monday mornings when it rains",
        "you are amazing at writing python code"
    ];

    // Language Model Implementation
    class NGramLanguageModel {
        constructor(order = 3, k = 1.0) {
            this.order = order;
            this.k = k;
            this.vocab = new Set();
            this.ngramCounts = {};
            this.prevCounts = {};
            this.train();
        }

        // Tokenize text into words and lowercase, removing extra spacing
        tokenize(text) {
            return text.toLowerCase().match(/\b\w+\b|[.!?]/g) || [];
        }

        // Build N-grams count matrices
        train() {
            this.vocab.clear();
            this.ngramCounts = {};
            this.prevCounts = {};

            // Add standard starting/ending tokens to vocab
            this.vocab.add("<s>");
            this.vocab.add("<e>");

            // Extract vocabulary and build N-grams
            corpus.forEach(sentence => {
                const words = this.tokenize(sentence);
                
                // Add words to vocabulary
                words.forEach(w => this.vocab.add(w));

                // Process up to max order (6-grams)
                for (let n = 1; n <= 6; n++) {
                    if (!this.ngramCounts[n]) {
                        this.ngramCounts[n] = {};
                        this.prevCounts[n] = {};
                    }

                    // Pad sentence with start and end tokens
                    const padStart = Array(n - 1).fill("<s>");
                    const padded = [...padStart, ...words, "<e>"];

                    for (let i = 0; i <= padded.length - n; i++) {
                        const ngram = padded.slice(i, i + n);
                        const prevNgram = ngram.slice(0, n - 1);
                        
                        const ngramKey = ngram.join(" ");
                        const prevKey = prevNgram.join(" ");

                        // Increment N-gram counts
                        this.ngramCounts[n][ngramKey] = (this.ngramCounts[n][ngramKey] || 0) + 1;
                        
                        // Increment preceding counts
                        this.prevCounts[n][prevKey] = (this.prevCounts[n][prevKey] || 0) + 1;
                    }
                }
            });
        }

        // Set parameters dynamically
        setHyperparameters(order, k) {
            this.order = order;
            this.k = k;
        }

        // Estimate probability for a word given the preceding n-gram context
        estimateProbability(word, prevTokens, n) {
            const vocabSize = this.vocab.size;
            const contextSize = n - 1;
            
            // Get relevant context tokens
            let context = [];
            if (prevTokens.length >= contextSize) {
                context = prevTokens.slice(-contextSize);
            } else {
                const pad = Array(contextSize - prevTokens.length).fill("<s>");
                context = [...pad, ...prevTokens];
            }

            const prevKey = context.join(" ");
            const fullKey = [...context, word].join(" ");

            const numCount = this.ngramCounts[n][fullKey] || 0;
            const denCount = this.prevCounts[n][prevKey] || 0;

            // Laplace k-smoothing estimation
            const probability = (numCount + this.k) / (denCount + this.k * vocabSize);
            return probability;
        }

        // Get predictions for a given input string
        getSuggestions(inputText, topK = 5) {
            const t0 = performance.now();
            const tokens = this.tokenize(inputText);
            const suggestions = [];

            // If text is empty or last char is space, we predict next word
            const isWordComplete = inputText.endsWith(" ") || inputText.length === 0;
            
            let contextTokens = [...tokens];
            let searchPrefix = "";

            if (!isWordComplete && tokens.length > 0) {
                // Last token is incomplete, use it as a filtering prefix
                searchPrefix = tokens[tokens.length - 1];
                contextTokens = tokens.slice(0, -1);
            }

            // Estimate probabilities for all words in vocabulary (excluding structural tokens)
            this.vocab.forEach(word => {
                if (word === "<s>" || word === "<e>") return;
                
                // If user is half-typing, filter words by prefix
                if (searchPrefix && !word.startsWith(searchPrefix)) return;
                
                // Avoid suggesting the exact input prefix itself if it is complete
                if (word === searchPrefix) return;

                // Estimate probability using chosen N-gram order
                const prob = this.estimateProbability(word, contextTokens, this.order);
                suggestions.push({ word, prob });
            });

            // Sort suggestions in descending order
            suggestions.sort((a, b) => b.prob - a.prob);

            const latency = performance.now() - t0;

            return {
                latency,
                results: suggestions.slice(0, topK)
            };
        }
    }

    // Initialize model instance with default Trigram (order=3) and k=1.0
    const model = new NGramLanguageModel(3, 1.0);

    // Update vocabulary diagnostics in DOM
    const updateDiagnostics = (latencyVal, vocabSize) => {
        document.getElementById("diag-latency").textContent = `${latencyVal.toFixed(3)} ms`;
        document.getElementById("diag-vocab").textContent = `${vocabSize} words`;
    };

    // ==========================================================================
    // 2. AUTO-COMPLETE INTERACTION MANAGEMENT (THE EDITOR CANVAS)
    // ==========================================================================

    const editor = document.getElementById("demo-input");
    const hintOverlay = document.getElementById("suggestion-hint-overlay");
    const suggestionsWrapper = document.getElementById("suggestions-wrapper");
    const presetsContainer = document.getElementById("presets-container");

    let currentSuggestions = [];
    let activeSuggestionIdx = 0;

    // Retrieve suggestions and update UI
    const processInput = () => {
        const text = editor.value;
        
        // Execute N-Gram recommendation query
        const response = model.getSuggestions(text, 5);
        currentSuggestions = response.results;
        
        // Update latency metrics live
        updateDiagnostics(response.latency, model.vocab.size);

        // Update the suggestion tags container
        if (currentSuggestions.length === 0) {
            suggestionsWrapper.innerHTML = `<span style="color: var(--muted-color); font-size: 0.95rem; font-style: italic;">No suggestions available for this context.</span>`;
            hintOverlay.innerHTML = "";
            return;
        }

        // Render gorgeous tags with animation
        suggestionsWrapper.innerHTML = currentSuggestions.map((item, idx) => {
            const percentage = (item.prob * 100).toFixed(1);
            return `
                <button class="suggestion-tag ${idx === activeSuggestionIdx ? 'active-selection' : ''}" data-word="${item.word}" data-index="${idx}">
                    ${item.word}
                    <span class="prob">${percentage}%</span>
                </button>
            `;
        }).join("");

        // Setup click events on new suggestion buttons
        document.querySelectorAll(".suggestion-tag").forEach(tag => {
            tag.addEventListener("click", (e) => {
                const word = e.currentTarget.getAttribute("data-word");
                autocompleteWord(word);
            });
        });

        // Update inline ghost text completion hint
        renderGhostCompletion(text);
    };

    // Render ghost text previewing the top suggestion
    const renderGhostCompletion = (text) => {
        if (currentSuggestions.length === 0 || text.length === 0) {
            hintOverlay.innerHTML = "";
            return;
        }

        const topWord = currentSuggestions[0].word;
        const tokens = model.tokenize(text);
        const endsWithSpace = text.endsWith(" ");

        let completionText = "";

        if (endsWithSpace) {
            // Suggesting next word entirely
            completionText = topWord;
            hintOverlay.innerHTML = `${text}<span class="autocomplete-hint">${completionText}</span>`;
        } else if (tokens.length > 0) {
            // Suggesting completion of the current prefix
            const lastToken = tokens[tokens.length - 1];
            if (topWord.startsWith(lastToken)) {
                completionText = topWord.slice(lastToken.length);
                hintOverlay.innerHTML = `${text}<span class="autocomplete-hint">${completionText}</span>`;
            } else {
                hintOverlay.innerHTML = "";
            }
        } else {
            hintOverlay.innerHTML = "";
        }
    };

    // Append completed word to text area
    const autocompleteWord = (word) => {
        const text = editor.value;
        const tokens = model.tokenize(text);
        const endsWithSpace = text.endsWith(" ");

        let newText = "";

        if (endsWithSpace || text.length === 0) {
            newText = text + word + " ";
        } else if (tokens.length > 0) {
            // Replace incomplete prefix with the word
            const lastToken = tokens[tokens.length - 1];
            newText = text.substring(0, text.length - lastToken.length) + word + " ";
        }

        editor.value = newText;
        editor.focus();
        
        // Re-process suggestions for the new context
        activeSuggestionIdx = 0;
        processInput();
    };

    // Keyboard controls for the editor: TAB key completion, Arrows navigation
    editor.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
            // Stop browser default focus shift
            e.preventDefault();
            
            if (currentSuggestions.length > 0) {
                // Complete word with the currently selected suggestion
                const wordToComplete = currentSuggestions[activeSuggestionIdx].word;
                autocompleteWord(wordToComplete);
            }
        } else if (e.key === "ArrowRight" && editor.selectionStart === editor.value.length) {
            // ArrowRight completes when cursor is at the very end
            if (currentSuggestions.length > 0) {
                e.preventDefault();
                autocompleteWord(currentSuggestions[0].word);
            }
        } else if (e.key === "ArrowDown") {
            // Navigate suggestions with arrow down
            if (currentSuggestions.length > 1) {
                e.preventDefault();
                activeSuggestionIdx = (activeSuggestionIdx + 1) % currentSuggestions.length;
                processInput();
            }
        } else if (e.key === "ArrowUp") {
            // Navigate suggestions with arrow up
            if (currentSuggestions.length > 1) {
                e.preventDefault();
                activeSuggestionIdx = (activeSuggestionIdx - 1 + currentSuggestions.length) % currentSuggestions.length;
                processInput();
            }
        }
    });

    // Run suggestion engine on every keystroke
    editor.addEventListener("input", () => {
        activeSuggestionIdx = 0;
        processInput();
    });

    // Mirror scrolling between textarea and overlay to keep background elements perfectly aligned
    editor.addEventListener("scroll", () => {
        hintOverlay.scrollTop = editor.scrollTop;
        hintOverlay.scrollLeft = editor.scrollLeft;
    });

    // Curated context presets triggers
    presetsContainer.addEventListener("click", (e) => {
        if (e.target.classList.contains("preset-btn")) {
            const prompt = e.target.getAttribute("data-text");
            editor.value = prompt + " ";
            editor.focus();
            activeSuggestionIdx = 0;
            processInput();
        }
    });

    // ==========================================================================
    // 3. TUNING AND HYPERPARAMETERS MANAGEMENT
    // ==========================================================================

    const orderSelect = document.getElementById("order-select");
    const orderValDisplay = document.getElementById("order-val");
    const smoothingSlider = document.getElementById("smoothing-slider");
    const smoothingValDisplay = document.getElementById("smoothing-val");

    // Dynamic model adjustments on order selection change
    orderSelect.addEventListener("change", (e) => {
        const order = parseInt(e.target.value);
        
        // Update styling indicator
        const orderLabels = ["Unigram", "Bigram", "Trigram", "Quadgram", "Quintgram", "Sextgram"];
        orderValDisplay.textContent = `${order}-Gram (${orderLabels[order - 1]})`;
        
        // Reconfigure model order
        model.setHyperparameters(order, parseFloat(smoothingSlider.value));
        processInput();
    });

    // Dynamic smoothing adjustments on slider slide
    smoothingSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value).toFixed(1);
        smoothingValDisplay.textContent = val;
        
        // Reconfigure model smoothing
        model.setHyperparameters(parseInt(orderSelect.value), parseFloat(val));
        processInput();
    });

    // Initialize display with default metrics
    processInput();

    // ==========================================================================
    // 4. SEAMLESS SDK INTEGRATION - TABBED INTERFACE
    // ==========================================================================

    const integrationTabs = document.getElementById("integration-tabs-list");
    const codeSnippetBox = document.getElementById("code-snippet-box");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const codeTitleDisplay = document.getElementById("code-title-display");

    integrationTabs.addEventListener("click", (e) => {
        const tabBtn = e.target.closest(".usage-tab-btn");
        if (!tabBtn) return;

        // Toggle active button
        document.querySelectorAll(".usage-tab-btn").forEach(btn => btn.classList.remove("active"));
        tabBtn.classList.add("active");

        // Toggle active tab pane
        const targetTabId = tabBtn.getAttribute("data-tab");
        tabPanes.forEach(pane => {
            pane.classList.remove("active");
            if (pane.id === targetTabId) {
                pane.classList.add("active");
            }
        });

        // Update display titles based on content category
        const titles = {
            "tab-installation": "bash - installation commands",
            "tab-training": "python - training n-gram parameters",
            "tab-inference": "python - pipeline execution & predictions"
        };
        codeTitleDisplay.textContent = titles[targetTabId];
    });

    // ==========================================================================
    // 5. ANIMATIONS & GRAPHIC INTERACTION
    // ==========================================================================

    // Dynamic Shrink on scroll navigation menu
    const nav = document.getElementById("main-nav");
    window.addEventListener("scroll", () => {
        if (window.scrollY > 40) {
            nav.classList.add("scrolled");
        } else {
            nav.classList.remove("scrolled");
        }
        
        // Dynamic scroll menu link highlights
        highlightActiveSection();
    });

    // Highlight menu items contextually relative to scroll viewport heights
    const sections = document.querySelectorAll("section");
    const navLinks = document.querySelectorAll(".nav-link");

    const highlightActiveSection = () => {
        let scrollPosition = window.scrollY + 180;

        sections.forEach(section => {
            if (scrollPosition >= section.offsetTop && scrollPosition < (section.offsetTop + section.offsetHeight)) {
                const currentId = section.getAttribute("id");
                
                navLinks.forEach(link => {
                    link.classList.remove("active");
                    if (link.getAttribute("href") === `#${currentId}`) {
                        link.classList.add("active");
                    }
                });
            }
        });
    };

    // Mobile Navbar Slide-out trigger
    const mobileToggle = document.getElementById("mobile-toggle");
    const navMenu = document.getElementById("nav-menu");

    mobileToggle.addEventListener("click", () => {
        const isOpen = navMenu.classList.toggle("open");
        mobileToggle.classList.toggle("open");
        mobileToggle.setAttribute("aria-expanded", isOpen);
    });

    // Close menu when mobile links are clicked
    navMenu.addEventListener("click", (e) => {
        if (e.target.classList.contains("nav-link")) {
            navMenu.classList.remove("open");
            mobileToggle.classList.remove("open");
            mobileToggle.setAttribute("aria-expanded", false);
        }
    });

    // Smooth entry animations for scroll viewport triggers (Fade + Slide Up)
    const animElements = document.querySelectorAll(".fade-slide-up");
    const animObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("appear");
                animObserver.unobserve(entry.target); // Trigger only once
            }
        });
    }, {
        threshold: 0.15
    });

    animElements.forEach(el => animObserver.observe(el));

    // Perplexity Chart Animation Trigger
    const chartViewport = document.getElementById("chart-viewport");
    const chartObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                chartViewport.classList.add("active");
                chartObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.3
    });

    if (chartViewport) {
        chartObserver.observe(chartViewport);
    }

    // ==========================================================================
    // 6. CONTACT INQUIRY DUMMY SUBMISSION ANIMATION
    // ==========================================================================

    const inquiryForm = document.getElementById("inquiry-form");
    if (inquiryForm) {
        inquiryForm.addEventListener("submit", (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById("form-submit-btn");
            const originalText = submitBtn.textContent;
            
            // Set dynamic loading transitions
            submitBtn.textContent = "Processing Inquiries...";
            submitBtn.style.opacity = "0.7";
            submitBtn.disabled = true;

            setTimeout(() => {
                // Success State transitions
                submitBtn.textContent = "Sent Successfully ✓";
                submitBtn.style.backgroundColor = "#4ab36a";
                submitBtn.style.color = "#ffffff";
                submitBtn.style.opacity = "1";
                
                // Clear the form fields
                inquiryForm.reset();
                
                // Return button to default state after visual cooldown
                setTimeout(() => {
                    submitBtn.textContent = originalText;
                    submitBtn.style.backgroundColor = "";
                    submitBtn.style.color = "";
                    submitBtn.disabled = false;
                }, 4000);
            }, 1800);
        });
    }
});
