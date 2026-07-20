import { useState, useEffect } from 'react';
import { ChevronUp } from 'lucide-react';

export default function ScrollTop() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const toggleVisibility = () => {
      setIsVisible(window.scrollY > 300);
    };

    window.addEventListener('scroll', toggleVisibility, { passive: true });
    return () => window.removeEventListener('scroll', toggleVisibility);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <button
      className={`scroll-top ${isVisible ? 'visible' : ''}`}
      id="scroll-top"
      aria-label="Scroll to top"
      onClick={scrollToTop}
    >
      <ChevronUp className="icon" size={24} />
    </button>
  );
}