import { render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { App } from '../App'

export function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}
