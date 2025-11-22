import React, { useState, useRef } from "react";
import icon from "../assets/csv-file-icon.png"; // เปลี่ยนเป็น path icon ของคุณ
import upload from "../assets/upload.png";
import Button from "./ui/Button";
import { Loader2, Send } from "lucide-react";


export default function CsvUploadCard() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) setFile(selectedFile);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) setFile(droppedFile);
  };

  const processCsv = () => {
    if (!file) return alert("Please upload a CSV file first!");
    console.log("Processing CSV:", file);
    // TODO: ส่งไป backend ของคุณเลย เช่นใช้ FormData
  };

  return (

    <>
      <div className="col-start-3 col-end-11 ">
        <div className="flex flex-col  bg-white shadow-lg rounded-xl p-8 space-y-4">
          <div className="flex items-center space-x-3 mb-4 ">
            <img src={icon} alt="CSV Icon" className="size-10" />
            <h1 className="text-2xl text-gray-800">Batch Upload</h1>
          </div>

      {/* Drag & Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-7 text-center cursor-pointer transition
          ${isDragging ? "border-indigo-500 bg-indigo-50" : "border-gray-300"}
        `}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="flex flex-col items-center justify-center space-y-2 h-44">
          <img src={upload} alt="" className="size-10"/>

          {!file && (
            <div>
              <p className="text-gray-500">Drag & drop your CSV file here</p>
              <p className="text-gray-400 text-sm">or click to browse</p>
            </div>
          )}

          {file && (
            <p className="text-gray-700 font-medium">Click to change files</p>
          )}
        </div>
        </div>
        {file && (
            <div className="bg-gray-100 flex items-center space-x-4 p-4 rounded-lg mt-4 text-black"> 
            <img src={icon} alt="" className="size-6" />
            <p> {file?.name ?? "No file selected"} </p>
        </div>
        )}
       
        {/* Hidden file input */}
      <input
        type="file"
        accept=".csv"
        ref={fileInputRef}
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Button */}
      <Button
                        onClick={processCsv}
                        disabled={!file || isAnalyzing}
                    > 
                        {isAnalyzing ? (
                            <div className="flex justify-center items-center space-x-2 ">
                            <Loader2 className="h-4 w-4 animate-spin " />
                            <p>Analyzing csv file...</p>
                            </div>
                        ) : (
                            <div className="flex justify-center items-center space-x-2">
                            <Send className="h-4 w-4" />
                            <p>Analyze csv file</p>
                            </div>
                        )}

                    </Button>
        
      </div>

      
      </div>
        {/* Header */}
      
    </>
    
  );
}
