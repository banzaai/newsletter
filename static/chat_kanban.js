document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("searchForm");
    const input = document.getElementById("queryInput");
    const resultContainer = document.getElementById("resultContainer");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = input.value.trim();

        if (!query) {
            resultContainer.innerHTML = "<p>Please enter a query.</p>";
            return;
        }

        try {
            const response = await fetch(`/api/kanban_query?event=${encodeURIComponent(query)}`);

            const data = await response.json();

const content = data.results?.map(item => `<li>${item}</li>`).join('') || "No results found.";
resultContainer.innerHTML = `<h3>Result:</h3><ul>${content}</ul>`;


        } catch (error) {
            console.error("Error fetching data:", error);
            resultContainer.innerHTML = "<p>Error retrieving data.</p>";
        }
    });
});
