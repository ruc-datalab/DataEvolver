import { useEffect, useState } from 'react'
import { MainLayout } from './layouts/MainLayout'
import { LandingPage } from './pages/LandingPage'
import { Toast } from './components/Toast'
import { useAppStore } from './stores/appStore'

export default function App() {
  const toast = useAppStore((s) => s.toast)
  const setToast = useAppStore((s) => s.setToast)
  const themeMode = useAppStore((s) => s.themeMode)
  const [showLanding, setShowLanding] = useState(true)

  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('theme-light', 'theme-dark')
    root.classList.add(themeMode === 'light' ? 'theme-light' : 'theme-dark')
  }, [themeMode])

  return (
    <>
      {showLanding
        ? <LandingPage onEnter={() => setShowLanding(false)} />
        : <MainLayout />
      }
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </>
  )
}