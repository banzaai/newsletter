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

        // Show loading message placeholder
        const loadingMsg = appendMessage("assistant", "Thinking...");

        try {
            const response = await fetch(`/api/kanban_query?query=${encodeURIComponent(query)}`);
            const msg = await response.json();
            console.log("msg:", msg);

            // Determine the response content
            let content = "";

            if (msg && typeof msg === "object" && msg.html !== undefined) {
                content = msg.html;
            } else if (typeof msg === "string") {
                content = msg;
            } else {
                content = "<em>Empty or invalid response</em>";
            }

            // Replace the placeholder with full HTML content
            loadingMsg.innerHTML = `
                <strong>Assistant:</strong><br>
                <div class="assistant-response">${content}</div>
            `;

        } catch (error) {
            console.error("Error rendering message:", error);
            loadingMsg.innerHTML = `<strong>System:</strong> Error retrieving data.`;
        }
    });

    function appendMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("chat-bubble", sender);

        // If assistant, render block container
        if (sender === "assistant") {
            messageDiv.innerHTML = `
                <strong>${formatSender(sender)}:</strong><br>
                <div class="assistant-response">${message}</div>
            `;
        } else {
            messageDiv.innerHTML = `<strong>${formatSender(sender)}:</strong> ${message}`;
        }

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
