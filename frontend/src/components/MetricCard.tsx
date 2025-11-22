

export default function MetricCards() {
    return(
         <div className="col-start-3 col-end-11 row-span-3 space-y-4 self-center">
        <div className="flex space-x-4  ">
            <div className="flex flex-col w-1/4 justify-center  bg-white rounded-xl space-y-2 p-4">
                <p >Accuracy</p>
                <p className="text-2xl font-semibold"> 90.23%</p>
            </div>
            <div className="flex flex-col w-1/4 justify-center  bg-white rounded-xl space-y-2 p-4">
                <p >Precision</p>
                <p className="text-2xl font-semibold"> 90.32%</p>
            </div>
            <div className="flex flex-col w-1/4 justify-center  bg-white rounded-xl space-y-2 p-4">
                <p >Recall</p>
                <p className="text-2xl font-semibold"> 90.43%</p>
            </div>
            <div className="flex flex-col w-1/4 justify-center  bg-white rounded-xl space-y-2 p-4">
                <p >F1 Score</p>
                <p className="text-2xl font-semibold"> 90.34%</p>
            </div>
        </div>
        
      </div>
    )
}