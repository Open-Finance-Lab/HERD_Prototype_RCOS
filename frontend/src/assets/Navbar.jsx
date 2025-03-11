import { useState } from 'react'
import './navbar.css'

const Navbar = () =>{
  return (
    <>
      <div id="navbar">
        <div id="navbarLeft">
          <a className="active" href="#">Home</a>
          <a href="#">About</a>
          <a href="#">Chat</a>
        </div>

        <div id="navbarRight">
          <a href="#">Sign In</a>
        </div>

      </div>
    </>
  )
}

export default Navbar;