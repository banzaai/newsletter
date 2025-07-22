from langchain.tools import tool
import json
import matplotlib.pyplot as plt
import tempfile
import os
import uuid

filename = None

@tool
def generate_plot(data_json: str) -> dict:
    """
    Generates a plot from JSON data and saves it to a temporary file.
    Accepts optional configuration for plot type and styling.
    Returns a dictionary with the filename and a short description.
    """

    try:
        data = json.loads(data_json)
        x = data.get("x")
        y = data.get("y")
        config = data.get("config", {})

        if not x or not y:
            return {"error": "Missing 'x' or 'y' in data."}
        if len(x) != len(y):
            return {"error": "'x' and 'y' must be the same length."}

        plot_type = config.get("plot_type", "line")
        color = config.get("color", None)
        title = config.get("title", "")
        xlabel = config.get("xlabel", "")
        ylabel = config.get("ylabel", "")
        grid = config.get("grid", False)
        figsize = tuple(config.get("figsize", (6, 4)))

        plt.figure(figsize=figsize)

        if plot_type == "line":
            plt.plot(x, y, color=color)
        elif plot_type == "bar":
            plt.bar(x, y, color=color)
        elif plot_type == "scatter":
            plt.scatter(x, y, color=color)
        elif plot_type == "hist":
            plt.hist(y, bins=config.get("bins", 10), color=color)
        elif plot_type == "pie":
            plt.pie(y, labels=x, autopct='%1.1f%%')
        else:
            return {"error": f"Unsupported plot type: {plot_type}"}

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid:
            plt.grid(True)

        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(tempfile.gettempdir(), filename)
        plt.savefig(file_path)
        plt.close()

        return {
            "text": f"{plot_type.capitalize()} plot saved.",
            "filename": filename,
            "path": file_path,
            "description": f"A {plot_type} plot with {len(x)} data points."
        }

    except Exception as e:
        return {"error": str(e)}
