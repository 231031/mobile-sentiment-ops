// DownloadButton.tsx (Component ใหม่)

import React from 'react';
import { Download } from 'lucide-react'; // ใช้ Icon สำหรับดาวน์โหลด

interface DownloadButtonProps {
    csvData: string | null; // ข้อมูล CSV ที่เป็น String
    fileName?: string; // ชื่อไฟล์สำหรับดาวน์โหลด (ตัวเลือก)
}

const DownloadButton: React.FC<DownloadButtonProps> = ({ csvData, fileName = 'analysis_result.csv' }) => {
    

    const handleDownload = () => {
        if (!csvData) {
            alert("No CSV data available to download.");
            return;
        }


        const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
        
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName; 
        
        link.click();
        URL.revokeObjectURL(url);
    };

    return (
        <button
            onClick={handleDownload}
            disabled={!csvData} 
            className={`
                bg-green-600 hover:bg-green-600 text-white font-medium py-2 px-4 rounded-lg 
                flex items-center space-x-2 transition disabled:bg-gray-500 cursor-pointer
            `}
        >
            <Download className="h-4 w-4" />
            <span>Download CSV </span>
        </button>
    );
};

export default DownloadButton;