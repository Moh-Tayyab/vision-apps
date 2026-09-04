# SECURITY.md — Security Considerations

## Authentication
- **Status**: No authentication implemented
- **Risk**: Anyone on the network can access the dashboard and control camera
- **Mitigation**: Deploy on isolated network, use firewall rules

## Model File Protection
- YOLO model file (.pt) stored in project root
- Model file should be in .gitignore (large file)
- Environment variable YOLO_MODEL_PATH can override default path

## Network Security
- CORS allows all origins (allow_origins=["*"])
- **Risk**: Cross-origin requests from malicious sites
- **Mitigation**: Restrict CORS in production deployment

## Input Validation
- File uploads validated for non-empty content
- Image decoding validates format
- Query parameters have bounds checking (confidence 0.05-0.95, etc.)

## Data Exposure
- No sensitive data stored or logged
- API key never exposed in responses
- Frame data processed in-memory only

## Recommendations for Production
- Add authentication middleware
- Restrict CORS to specific origins
- Add rate limiting to API endpoints
- Use HTTPS for all connections
- Add request logging for audit trail
