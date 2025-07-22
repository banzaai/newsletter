document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("searchForm");
    const input = document.getElementById("queryInput");
    const bencherInput = document.getElementById("bencherInput");
    const incompleteCheckbox = document.getElementById("incompleteOnly");
    const chatWindow = document.getElementById("chatWindow");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        const bencher = bencherInput?.value.trim();
        const incompleteOnly = incompleteCheckbox?.checked;

        // Only proceed if there's a query or at least one filter
        if (!query && !bencher && !incompleteOnly) {
            appendMessage("system", "Please enter a query or select a filter.");
            return;
        }

        if (query) appendMessage("user", query);
        input.value = "";

        const loadingMsg = appendMessage("assistant", "Thinking...");

        try {
            const queryParams = new URLSearchParams();
            let endpoint = "/api/kanban_query";

        if (bencher || incompleteOnly) {
            endpoint = "/api/kanban_query_filtered";
            if (query) queryParams.append("query", query);
            if (bencher) queryParams.append("bencher", bencher);
            if (incompleteOnly) queryParams.append("incomplete_only", "true");
        } else {
            queryParams.append("query", query);
        }


            console.log("Calling:", endpoint, "with:", queryParams.toString());

            const response = await fetch(`${endpoint}?${queryParams.toString()}`);
            const msg = await response.json();

            if (msg && typeof msg === "object" && msg['filename'] != null) {
                const imageUrl = `/api/plot?filename=${encodeURIComponent(msg['filename'])}`;
                loadingMsg.innerHTML = `
                    <strong>Assistant:</strong><br>
                    <img src="${imageUrl}" alt="Generated Plot" style="max-width: 100%; height: auto;" /><br>
                    <p>${msg['html'] || ""}</p>
                `;
            } else {
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
