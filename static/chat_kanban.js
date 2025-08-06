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

    const rawHtml = msg.html || "";
    let chartUrl = "";

    // Create a temporary DOM element to parse the HTML
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = rawHtml;

    // Find the first <a> tag and extract the href
    const link = tempDiv.querySelector("a");
    if (link && link.href) {
        const href = link.getAttribute("href");
        const chartPathIndex = href.indexOf("/api/chart/");
        if (chartPathIndex !== -1) {
            chartUrl = href.slice(chartPathIndex); // removes everything before /chart/
        }
        console.log("Chart URL found:", chartUrl);
    }

    // Render the assistant response
    loadingMsg.innerHTML = `
        <strong>Assistant:</strong><br>
        <div class="assistant-response">${rawHtml}</div>
    `;

    if (chartUrl) {
        // Wait 500ms before embedding the chart
        setTimeout(() => {
            const container = loadingMsg.querySelector(".assistant-response");
            if (container) {
                const iframe = document.createElement("iframe");
                iframe.src = chartUrl;
                iframe.width = "100%";
                iframe.height = "400px";
                iframe.style.border = "none";
                container.appendChild(iframe);
            } else {
                console.warn("Could not find .assistant-response container to append chart iframe.");
            }
        }, 2000);
    } else {
        // fallback if no chart URL found
        loadingMsg.innerHTML = `
            <strong>Assistant:</strong><br>
            <div class="assistant-response">${rawHtml}</div>
        `;
    }

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
