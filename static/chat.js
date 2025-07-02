let currentMode = null;

function startChat(mode) {
    if (currentMode!=mode){
        document.getElementById("chatbox").innerHTML = "";
    }
    currentMode = mode;

    // Show the input area
    document.getElementById("inputArea").style.display = "block";

    // Update placeholder based on mode
    const contentInput = document.getElementById("contentInput");
    if (mode === "story") {
        contentInput.placeholder = "Your Story";
    } else if (mode === "event") {
        contentInput.placeholder = "Your Event";
    }

    // Highlight the selected button
    const storyBtn = document.querySelector("button[onclick=\"startChat('story')\"]");
    const eventBtn = document.querySelector("button[onclick=\"startChat('event')\"]");
    
    storyBtn.classList.remove("active");
    eventBtn.classList.remove("active");

    if (mode === "story") {
        storyBtn.classList.add("active");
    } else {
        eventBtn.classList.add("active");
    }
}

async function submitContent() {
    const name = document.getElementById("nameInput").value;
    const event = document.getElementById("contentInput").value;
    const chatbox = document.getElementById("chatbox");

    if (!name || !event || !currentMode) {
        alert("Please fill in all fields and select a mode.");
        return;
    }

    // Display user message
    const userMessage = document.createElement("div");
    userMessage.className = "message user";
    userMessage.textContent = `${name}: ${event}`;
    chatbox.appendChild(userMessage);

    // Send data to backend
    try {
        const response = await fetch(`/api/${currentMode}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ name, event })
        });


        const data = await response.json(); // Parse the JSON body
        const botReply = data.response; // Access the "response" field


        const botMessage = document.createElement("div");
        botMessage.className = "message bot";
        botMessage.textContent = botReply || "Received!";
        chatbox.appendChild(botMessage);
    } catch (error) {
        const errorMessage = document.createElement("div");
        errorMessage.className = "message bot";
        errorMessage.textContent = "Error contacting the server.";
        chatbox.appendChild(errorMessage);
        console.error("Error:", error);
    }

    // Scroll to bottom
    chatbox.scrollTop = chatbox.scrollHeight;

    // Clear inputs
    document.getElementById("contentInput").value = "";
}
