import icon from "../assets/text-box.png";
import Button from "./ui/Button";
import { Loader2, Send } from "lucide-react";
import { useState } from "react";

export default function TextFieldAnalysis() {

    const cards = [
        { id: 1, result: "Positive", color: "bg-green-400" },
        { id: 2, result: "Negative", color: "bg-red-400" },
        { id: 3, result: "Neutral", color: "bg-gray-400" },
    ];

    const [result, setResult] = useState<string>()
    const [text, setText] = useState<string>("")
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)

    const mappedCard = cards.find((c) => c.result === result)

    const analyzeSentiment = async () => {
        if (!text) return;

        setIsAnalyzing(true);

        try {
            const res = await fetch("/api/predict_json", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: text }),
            });

            if (!res.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await res.json();
            setResult(data.prediction);
        } catch (err) {
            console.error("Error analyzing text:", err);
            setResult("Error");
        } finally {
            setIsAnalyzing(false);
        }
    };

    return (
        <div className="col-start-3 col-end-11">
            <div className="flex flex-col bg-white shadow-lg rounded-xl p-8 space-y-4">
                <div className="flex mb-4 space-x-4 rounded-lg items-center">
                    <img src={icon} alt="" className="size-14" />
                    <div className="flex flex-col">
                        <h1 className="text-2xl">Text Analysis</h1>
                        <p className="text-gray-400">Enter text to analyze sentiment</p>
                    </div>
                </div>

                <input
                    type="text"
                    placeholder="Type or paste your text here..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="bg-gray-100 w-full h-42 mb-4 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder-gray-400"
                />

                <Button
                    onClick={analyzeSentiment}
                    disabled={!text || isAnalyzing}
                >
                    {isAnalyzing ? (
                        <div className="flex justify-center items-center space-x-2">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <p>Analyzing...</p>
                        </div>
                    ) : (
                        <div className="flex justify-center items-center space-x-2">
                            <Send className="h-4 w-4" />
                            <p>Analyze</p>
                        </div>
                    )}
                </Button>

                {result && text && (
                    <div className="bg-gray-100 rounded-lg text-black p-4">
                        <div className="flex items-center">
                            <p>Sentiment</p>
                            <div className={`flex items-center w-24 h-6 justify-center text-sm ml-auto rounded-full text-white ${mappedCard?.color ?? "bg-gray-400"}`}>{mappedCard?.result ?? "Unknown"}</div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}