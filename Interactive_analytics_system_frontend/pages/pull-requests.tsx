import { GetServerSideProps } from 'next'
import Head from 'next/head'
import Link from 'next/link'

interface PullRequest {
  number: number
  title: string
  url: string
  created_at: string
  user: string
}

interface Props {
  pullRequests: PullRequest[]
  error: string | null
}

export default function PullRequestsPage({ pullRequests, error }: Props) {
  return (
    <>
      <Head>
        <title>Open Pull Requests – GAA Video Analysis</title>
      </Head>
      <div className="container">
        <h1>Open Pull Requests</h1>
        <p style={{ marginBottom: '20px' }}>
          <Link href="/" style={{ color: '#0070f3', textDecoration: 'underline' }}>
            ← Back to Video Analysis
          </Link>
        </p>

        {error && (
          <p style={{ color: 'red', marginBottom: '16px' }}>
            Failed to load pull requests: {error}
          </p>
        )}

        {!error && pullRequests.length === 0 && (
          <p>No open pull requests found.</p>
        )}

        {!error && pullRequests.length > 0 && (
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {pullRequests.map((pr) => (
              <li
                key={pr.number}
                style={{
                  borderBottom: '1px solid #eee',
                  padding: '14px 0',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span
                    style={{
                      background: '#2da44e',
                      color: 'white',
                      borderRadius: '12px',
                      padding: '2px 10px',
                      fontSize: '12px',
                      fontWeight: 600,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    Open
                  </span>
                  <a
                    href={pr.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: '#0070f3', fontWeight: 600, textDecoration: 'none' }}
                  >
                    #{pr.number} {pr.title}
                  </a>
                </div>
                <span style={{ fontSize: '13px', color: '#666' }}>
                  Opened by <strong>{pr.user}</strong> on{' '}
                  {new Date(pr.created_at).toLocaleDateString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  )
}

export const getServerSideProps: GetServerSideProps<Props> = async () => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  try {
    const res = await fetch(`${apiUrl}/pull-requests`)
    if (!res.ok) {
      const text = await res.text()
      return { props: { pullRequests: [], error: text } }
    }
    const pullRequests: PullRequest[] = await res.json()
    return { props: { pullRequests, error: null } }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return { props: { pullRequests: [], error: message } }
  }
}
