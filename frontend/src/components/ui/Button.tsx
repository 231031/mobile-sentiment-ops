import { useNavigate } from "react-router-dom";
import React from "react";


interface NavButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    label?: string;
    to?: string;
}


function Button({label, to, onClick, className, children, ...props}: NavButtonProps) {
    const navigate = useNavigate();

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
        if (onClick) {
            onClick(e)
        }

        if (to) {
            navigate(to)
        }
    }

    return (
        <button
            {...props}
            onClick={handleClick}
            className={`w-full bg-orange-400 hover:bg-blue-500 text-white py-2 rounded-lg font-medium cursor-pointer ${className}`}
        >
            {children ?? label}
        </button>
    )
}


export default Button;