import { useState } from 'react'
import type { VideoMetadata } from '../types'
import { uploadVideo } from '../lib/api'

interface VideoUploaderProps {
  onUploadSuccess: (metadata: VideoMetadata, file: File) => void
}

export default function VideoUploader({ onUploadSuccess }: VideoUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadingVideo, setUploadingVideo] = useState(false)
  const [uploadedMetadata, setUploadedMetadata] = useState<VideoMetadata | null>(null)
  const [error, setError] = useState('')

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setUploadedMetadata(null)
      setError('')
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setUploadingVideo(true)
    setError('')
    try {
      const metadata = await uploadVideo(selectedFile)
      setUploadedMetadata(metadata)
      onUploadSuccess(metadata, selectedFile)
    } catch (err: any) {
      setError(err.message || 'Failed to upload video')
    } finally {
      setUploadingVideo(false)
    }
  }

  return (
    <div className="upload-section">
      <h2>1. Upload Video</h2>
      <div className="file-input">
        <input type="file" accept="video/mp4" onChange={handleFileChange} />
      </div>
      {selectedFile && !uploadedMetadata && (
        <div>
          <p>Selected: {selectedFile.name}</p>
          <button onClick={handleUpload} disabled={uploadingVideo}>
            {uploadingVideo ? 'Uploading...' : 'Upload & Analyze'}
          </button>
        </div>
      )}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {uploadedMetadata && (
        <div className="video-info">
          <h3>Video Information</h3>
          <div className="info-grid">
            <div><strong>Duration:</strong> {uploadedMetadata.duration_seconds.toFixed(2)}s</div>
            <div><strong>FPS:</strong> {uploadedMetadata.fps}</div>
            <div><strong>Total Frames:</strong> {uploadedMetadata.num_frames}</div>
            <div><strong>Resolution:</strong> {uploadedMetadata.width} x {uploadedMetadata.height}</div>
          </div>
        </div>
      )}
    </div>
  )
}
