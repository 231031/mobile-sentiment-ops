import { useState, useEffect } from "react"

export default function MetricCards() {

    const [metrics, setMetrics] = useState([
        { id: 1, name: "Accuracy", value: "-" },
        { id: 2, name: "Precision", value: "-" },
        { id: 3, name: "Recall", value: "-" },
        { id: 4, name: "F1 Score", value: "-" },
    ])

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const res = await fetch(`${import.meta.env.VITE_API_URL}/model/metrics`);
                if (!res.ok) throw new Error("Failed to fetch metrics");
                const data = await res.json();

                console.log(data)

                // Helper to format percentage
                const fmt = (val: number) => (val * 100).toFixed(2) + "%";

                setMetrics([
                    { id: 1, name: "Accuracy", value: data.accuracy ? fmt(data.accuracy) : "-" },
                    { id: 2, name: "Precision", value: data.macro_precision ? fmt(data.macro_precision) : "-" },
                    { id: 3, name: "Recall", value: data.macro_recall ? fmt(data.macro_recall) : "-" },
                    { id: 4, name: "F1 Score", value: data.micro_f1 ? fmt(data.micro_f1) : "-" },
                    
                ]);
            } catch (err) {
                console.error("Error fetching metrics:", err);
            }
        };

        fetchMetrics();
    }, []);

    return (
        <div className="col-start-3 col-end-11 row-span-3 space-y-4 self-center">
            <div className="flex space-x-4  ">

                {metrics.map((metric) => (
                    <div key={metric.id} className="flex flex-col w-1/4 justify-center  bg-white rounded-xl space-y-2 p-4">
                        <p >{metric.name}</p>
                        <p className="text-2xl font-semibold"> {metric.value}</p>
                    </div>
                ))}
            </div>

        </div>
    )
}