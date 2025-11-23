
import { useState } from "react"

export default function MetricCards() {

    const [metrics, setMetrics] = useState([
        {id: 1, name: "Accuracy", value: "90.23%" },
        {id: 2, name: "Precision", value: "90.32%" },
        {id: 3, name: "Recall", value: "90.43%" },
        {id: 4, name: "F1 Score", value: "90.34%" },
    ])
    
    return(


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