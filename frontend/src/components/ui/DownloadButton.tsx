// DownloadButton.tsx (Component ใหม่)

import React from 'react';
import { Download } from 'lucide-react'; // ใช้ Icon สำหรับดาวน์โหลด

interface DownloadButtonProps {
    csvData: string | null; // ข้อมูล CSV ที่เป็น String
    fileName?: string; // ชื่อไฟล์สำหรับดาวน์โหลด (ตัวเลือก)
}

const DownloadButton: React.FC<DownloadButtonProps> = ({ csvData, fileName = 'analysis_result.csv' }) => {
    
    // ฟังก์ชันที่จะถูกเรียกเมื่อปุ่มถูกคลิก
    const handleDownload = () => {
        if (!csvData) {
            alert("No CSV data available to download.");
            return;
        }

        // 1. สร้าง Blob จากข้อมูล CSV ที่เป็น String
        const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
        
        // 2. สร้าง URL สำหรับ Blob
        const url = URL.createObjectURL(blob);
        
        // 3. สร้าง element <a> แบบซ่อนและตั้งค่า
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName; // ใช้ชื่อไฟล์ที่กำหนด
        
        // 4. ทริกเกอร์การคลิกเพื่อเริ่มการดาวน์โหลด
        link.click();

        // 5. ทำความสะอาด (ลบ link และ revoke URL)
        URL.revokeObjectURL(url);
    };

    return (
        // ใช้ Button Component เดิมของคุณ หรือสร้างปุ่มใหม่
        <button
            onClick={handleDownload}
            disabled={!csvData} // ปิดปุ่มถ้าไม่มีข้อมูล
            className={`
                bg-green-500 hover:bg-green-600 text-white font-medium py-2 px-4 rounded-lg 
                flex items-center space-x-2 transition disabled:bg-gray-400
            `}
        >
            <Download className="h-4 w-4" />
            <span>Download CSV </span>
        </button>
    );
};

export default DownloadButton;