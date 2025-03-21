import { useState } from 'react'
import "./About.css"
import Main from '../assets/Main'


function About() {


  return (
    <>
      <Main>
        <div id='missionStatement'>
          <div id="missionStatementBg">
            <div id="missionStatementImg">
              <img className="cardImgMission" src="public/placeHolderImg.png" alt="Content Image" />
            </div>
            <div id="missionStatementText">
              <h1>Mission Statement</h1>
            </div>
          </div>
        </div>
      </Main>
    </>
  )
}

export default About
  