import { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';

const navLinks = [
  { path: '/#about', label: 'About' },
  { path: '/#education', label: 'Education' },
  { path: '/#projects', label: 'Projects' },
  { path: '/#skills', label: 'Skills' },
  { path: '/#achievements', label: 'Achievements' },
  { path: '/#contact', label: 'Contact' },
];

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  const closeMenu = () => setIsMenuOpen(false);

  return (
    <header className={`header ${isScrolled ? 'scrolled' : ''}`} id="header">
      <nav className="nav container">
        <NavLink to="/" className="nav__logo" onClick={closeMenu}>
          [Surya K]
        </NavLink>
        <ul className={`nav__menu ${isMenuOpen ? 'active' : ''}`} id="nav-menu">
          {navLinks.map((link) => (
            <li key={link.path}>
              <NavLink
                to={link.path}
                className={({ isActive }) => `nav__link ${isActive ? 'active' : ''}`}
                onClick={closeMenu}
              >
                {link.label}
              </NavLink>
            </li>
          ))}
        </ul>
        <button
          className={`nav__toggle ${isMenuOpen ? 'active' : ''}`}
          id="nav-toggle"
          aria-label="Toggle navigation"
          onClick={toggleMenu}
        >
          <span className="nav__toggle-icon" />
        </button>
      </nav>
    </header>
  );
}