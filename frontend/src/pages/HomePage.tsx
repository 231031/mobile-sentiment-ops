
import csvIcon from "../assets/csv-file-icon.png";
import textIcon from "../assets/text-box.png";
import Button from "../components/ui/Button";
import MetricCards from "../components/MetricCard";


export default function Home() {
  return (
    <div className="h-screen bg-gray-100 grid grid-cols-12 grid-rows-12 gap-4 py-10 ">

      <div className="bg-white rounded-xl p-4 col-start-3 col-end-11 row-span-3">
        <div className="flex flex-col w-full h-full justify-center items-center space-y-4">
          <p className="text-4xl">Sentiment Analysis</p>
          <p>Final Project CPE393 </p>
        </div>
      </div>

      <MetricCards></MetricCards>
      
      <div className="bg-white rounded-xl p-8 col-start-3 col-span-4 row-span-5">
        <div className="flex flex-col w-full h-44 justify-start items-start space-y-4 ">
          <img src={textIcon} className="size-10 " />
          <p className="text-2xl">Text Analysis</p>
          <p className="text-gray-400"> text input for real-time sentiment prediction</p>
          <Button label=" Analyze Text " to="/text-analysis" className="w-full"/>
        </div>
      </div>

      <div className="bg-white rounded-xl p-8 col-start-7 col-span-4 row-span-5">
        <div className="flex flex-col w-full h-44 justify-start items-start space-y-4 ">
          <img src={csvIcon} className="size-10" />
          <p className="text-2xl">CSV Upload </p>
          <p className="text-gray-400"> Process large dataset with batch sentiment analysis</p>
          <Button label=" Upload CSV " to="/csv-upload" />
        </div>
      </div>


    </div>
  );
}
