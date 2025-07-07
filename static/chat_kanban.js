document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("searchForm");
    const input = document.getElementById("queryInput");
    const chatWindow = document.getElementById("chatWindow");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = input.value.trim();

        if (!query) {
            appendMessage("system", "Please enter a query.");
            return;
        }

        appendMessage("user", query);
        input.value = "";

        // Show loading message
        const loadingMsg = appendMessage("assistant", "Thinking...");
        
        try {
            const response = await fetch(`/api/kanban_query?query=${encodeURIComponent(query)}`);
            const data = await response.json();

            const content = typeof data === "string" ? data : JSON.stringify(data, null, 2);
            loadingMsg.innerHTML = `<strong>Assistant:</strong> ${content}`;
        } catch (error) {
            console.error("Error fetching data:", error);
            loadingMsg.innerHTML = `<strong>System:</strong> Error retrieving data.`;
        }
    });

    function appendMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("chat-bubble", sender);
        messageDiv.innerHTML = `<strong>${formatSender(sender)}:</strong> ${message}`;
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return messageDiv;
    }

    function formatSender(sender) {
        if (sender === "user") return "You";
        if (sender === "assistant") return "Assistant";
        return "System";
    }
});

