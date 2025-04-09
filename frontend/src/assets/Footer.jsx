import { useState } from 'react'
import './footer.css'

const Footer = () =>{
  // footer that holds the title, links to the home, about, and chat pages, and a link to the github repo
  return (
    <>
      <div className="footer">
        <div id="footerLeft">
            <div id="leftTop">
                <h3>HERD</h3>
                <div id="imgFooter">
                  <a href="https://github.com/YangletLiu/HERD_Prototype_RCOS"><img src="/githublogo.png" alt="GitHub Logo"></img></a>
                </div>
            </div>
        </div>
        <div id="footerMiddle">
            <a className="footerLink" href="#">Home</a>
            <p className="footerSeparators">|</p>
            <a className="footerLink" href="#">About</a>
            <p className="footerSeparators">|</p>
            <a className="footerLink" href="#">Chat</a>
        </div>
      </div>
    </>
  )
}

export default Footer;