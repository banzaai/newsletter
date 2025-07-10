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
            const msg = await response.json();
            console.log("msg.filename:", msg);

            if (msg && typeof msg === "object" && msg['filename'] != null) {

                const imageUrl = `/api/plot?filename=${encodeURIComponent(msg['filename'])}`;
                loadingMsg.innerHTML = `
                    <strong>Assistant:</strong><br>
                    <img src="${imageUrl}" alt="Generated Plot" style="max-width: 100%; height: auto;" /><br>
                    <p>${msg['html'] || ""}</p>
                `;
            } else {
                // Just a text response
                loadingMsg.innerHTML = `<strong>Assistant:</strong> ${typeof msg === "string" ? msg : msg['html']}`;
            }
        } catch (error) {
            console.error("Error rendering message:", error);
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

