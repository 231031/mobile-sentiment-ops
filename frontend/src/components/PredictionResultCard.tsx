

import {useState} from "react";

export default function PredictionResultCard() {

    const cards = [
        { id: 1, result: "Positive", color: "bg-green-400" },
        { id: 2, result: "Negative", color: "bg-red-400" },
        { id: 3, result: "Neutral", color: "bg-gray-400" },
    ];


    const [result, setResult] = useState("Positive");

    const matchedCard = cards.find((c) => c.result === result)

    return (
       <>
            <h1 className="text-2xl mb-4 text-gray-800">
                Prediction Result
            </h1>
            <div className={`${matchedCard.color} shadow-lg rounded-xl flex items-center justify-center h-60`}>
                {matchedCard.result}
             </div>
            
        </> 
        
    )
}