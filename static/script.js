const form = document.querySelector("#quoteForm");
const promptInput = document.querySelector("#prompt");
const generateBtn = document.querySelector("#generateBtn");
const quoteBox = document.querySelector("#generatedQuote");
const statusMessage = document.querySelector("#statusMessage");
const copyBtn = document.querySelector("#copyBtn");

let currentQuote = "";

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const prompt = promptInput.value.trim();

    if (!prompt) {
        statusMessage.textContent = "Please enter a prompt.";
        quoteBox.textContent = "";
        return;
    }

    generateBtn.disabled = true;
    generateBtn.textContent = "Generating...";
    statusMessage.textContent = "Generating your quote...";
    quoteBox.textContent = "";
    currentQuote = "";
    copyBtn.disabled = true;

    try {
        const response = await fetch("/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ prompt: prompt })
        });

        const data = await response.json();

        if (!response.ok || !data.success || !data.quote) {
            throw new Error(data.error || "No quote was generated.");
        }

        currentQuote = data.quote.trim();

        if (!currentQuote) {
            throw new Error("The generated quote is empty. Try another prompt.");
        }

        quoteBox.textContent = currentQuote;
        statusMessage.textContent = "Quote generated successfully!";
        copyBtn.disabled = false;

    } catch (error) {
        quoteBox.textContent = "";
        statusMessage.textContent = error.message;
        console.error("Generation error:", error);

    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate Quote";
    }
});

copyBtn.addEventListener("click", async function () {
    if (!currentQuote) {
        statusMessage.textContent = "Generate a quote before copying.";
        return;
    }

    try {
        await navigator.clipboard.writeText(currentQuote);
        statusMessage.textContent = "Quote copied successfully!";

    } catch (error) {
        // Fallback for browsers where Clipboard API is unavailable
        const textarea = document.createElement("textarea");
        textarea.value = currentQuote;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";

        document.body.appendChild(textarea);
        textarea.select();

        const copied = document.execCommand("copy");
        textarea.remove();

        statusMessage.textContent = copied
            ? "Quote copied successfully!"
            : "Copy failed. Please select and copy the quote manually.";
    }
});
const themeToggle = document.querySelector("#themeToggle");

themeToggle.addEventListener("click", function () {
    document.body.classList.toggle("light-mode");

    const isLight = document.body.classList.contains("light-mode");

    themeToggle.textContent = isLight
        ? "🌙 Switch to Dark Mode"
        : "☀️ Switch to Light Mode";
});