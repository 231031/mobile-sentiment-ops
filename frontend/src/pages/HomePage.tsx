
import csvIcon from "../assets/csv-file-icon.png";
import textIcon from "../assets/text-box.png";
import Button from "../components/ui/Button";

export default function Home() {
  return (
    <div className="h-screen bg-gray-100 grid grid-cols-12 grid-rows-12 gap-4 py-10 ">

      <div className="bg-white rounded-xl p-4 col-start-3 col-end-11 row-span-5">
        <div className="flex flex-col w-full h-full justify-center items-center space-y-4">
          <p className="text-4xl">Mobile Sentiment</p>
          <p>Final Project CPE393 </p>
        </div>
      </div>

      <div className="bg-white rounded-xl p-8 col-start-3 col-span-4 row-span-5" >
        <div className="flex flex-col w-full h-full justify-center items-start space-y-4  ">
          <img src={textIcon} className="size-10 " />
          <p className="text-2xl">Text Analysis</p>
          <p className="text-gray-400"> Text input for sentiment prediction</p>
          <Button label=" Analyze Text " to="/text-analysis" className="w-full"/>
        </div>
      </div>

      <div className="bg-white rounded-xl p-8 col-start-7 col-span-4 row-span-5">
        <div className="flex flex-col w-full h-full justify-center items-start space-y-4 ">
          <img src={csvIcon} className="size-10" />
          <p className="text-2xl">CSV Upload </p>
          <p className="text-gray-400"> CSV input for sentiment prediction</p>
          <Button label=" Upload CSV " to="/csv-upload" className="w-full"/>
        </div>
      </div>


    </div>
  );
}
