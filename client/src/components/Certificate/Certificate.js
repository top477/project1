import React, { useCallback, useEffect, useState } from 'react';
import jsPDF from 'jspdf';
import certificate from '../../assets/Images/certificate.png';
import Button from '../Button/Button';
import axios from 'axios';
import './style.css';
import config from '../../config/Apiconfig';

const Certificate = () => {
  const [name, setName] = useState('');
  const [eligible, setEligible] = useState(0);

  useEffect(() => {
    getName();
  }, [name]);

  const getName = async () => {
    const token = localStorage.getItem('token');
    const decoded = JSON.parse(atob(token.split(".")[1]));
    const response = await axios.get(`${config.baseURL}/${decoded.userId}`);
    setName(response.data.data.name);
    setEligible(response.data.data.progress.total);
  };

  const generatePDF = useCallback(() => {
    const pdf = new jsPDF({
      orientation: 'landscape',
      unit: 'px',
      format: [800, 550],
    });

    const img = new Image();
    img.src = certificate;
    img.onload = () => {
      pdf.addImage(img, 'PNG', 0, 0, 800, 550);
      pdf.setFont('times', 'italic');
      pdf.setFontSize(32);
      pdf.text(name, 400, 275, { align: 'center' });
      pdf.save('certificate.pdf');
    };
  }, [name]);

  return (
    <>
      {eligible === 100 ? (
        <div className='container'>
          <div className='certificate'>
            <img src={certificate} alt='Certificate' className='certificate-image' />
            <div className='name'>
              <h1>{name}</h1>
            </div>
          </div>
          <Button text="Download" onClick={generatePDF} />
        </div>
      ) : (
        <div className='wrap'>
          <h1>Please complete all tests in the Quiz section to get your certificate.</h1>
          <p>To unlock your certificate, you need to score 100% by completing all the tests available in the Quiz section. Keep up the hard work!</p>
        </div>
      )}
    </>
  );
};

export default Certificate;
