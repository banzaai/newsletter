document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("searchForm");
    const input = document.getElementById("queryInput");
    const typeInput = document.getElementById("queryType");
    const toggleBtn = document.getElementById("toggleTypeBtn");
    const resultContainer = document.getElementById("resultContainer");

    // Toggle between event and story
    toggleBtn.addEventListener("click", () => {
        const current = typeInput.value;
        const next = current === "event" ? "story" : "event";
        typeInput.value = next;
        toggleBtn.textContent = `Mode: ${next.charAt(0).toUpperCase() + next.slice(1)}`;
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        const type = typeInput.value;

        if (!query) {
            resultContainer.innerHTML = "<p>Please enter a query.</p>";
            return;
        }

        const endpoint = type === "event" ? "/event/" : "/story/";

        try {
            const response = await fetch(`api${endpoint}?query=${encodeURIComponent(query)}`);
            const data = await response.json();

            const content = data.events || data.stories || data.message || "No results found.";
            resultContainer.innerHTML = `
                <h3>Result:</h3>
                <p>${content}</p>
            `;
        } catch (error) {
            console.error("Error fetching data:", error);
            resultContainer.innerHTML = "<p>Error retrieving data.</p>";
        }
    });
});
