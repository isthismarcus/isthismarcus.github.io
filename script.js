class MarcusFaceRecognition {
    constructor() {
        this.session = null;
        this.faceDetection = null;
        this.isModelLoaded = false;
        this.isFaceDetectionLoaded = false;
        this.threshold = 0.2292; // Cosine similarity threshold for Marcus detection
        this.minConfidence = 36; // Minimum confidence percentage required
        
        // Marcus mean embedding from your training data
        this.marcusMeanEmbedding = [
            -0.00325362, -0.00142106, 0.00101414, -0.01034536, -0.00394643, -0.01844343, -0.01988791, 0.05717670,
            0.01061577, 0.05583358, -0.03058745, 0.00611985, -0.01577461, 0.02331577, -0.03944641, 0.09325164,
            0.00260496, -0.00001877, -0.01725520, -0.01403749, 0.00426589, -0.00062678, 0.04790544, -0.01550362,
            0.01687316, -0.02820176, -0.03313638, 0.00216808, 0.00582483, -0.00887210, -0.03592198, -0.01232490,
            0.00859034, -0.05625437, 0.03351146, 0.02547761, -0.00272476, -0.03800133, -0.02359386, -0.02196295,
            0.00958104, 0.04254368, -0.00638041, 0.02986733, 0.05474567, 0.04704892, -0.02407119, 0.01105702,
            -0.03227510, 0.04785699, 0.01618417, -0.06423647, -0.00770997, -0.05243413, -0.07880760, 0.03813281,
            -0.02303801, 0.00294838, 0.00109396, -0.00309520, 0.02280339, -0.05183254, -0.00924017, 0.05337931,
            -0.09720488, -0.00344404, 0.00169257, 0.03166025, 0.02198819, 0.00459826, -0.00123313, -0.02288858,
            0.04349195, 0.01754594, -0.01382595, -0.01921175, -0.02089978, 0.07739104, 0.06633127, 0.02003646,
            0.05509392, -0.02175873, -0.06943816, -0.00220973, -0.00763865, 0.03203453, 0.02420172, 0.08612281,
            -0.03479229, 0.01384271, -0.05457706, -0.05148736, 0.03008537, -0.04160416, -0.02813785, -0.02052059,
            0.00638607, 0.01012756, 0.00610130, 0.01810344, 0.04330646, 0.04561549, -0.02586697, 0.05232847,
            0.04208655, -0.02631154, -0.00133264, -0.03486652, 0.03648949, -0.05018891, 0.00325364, -0.03074316,
            0.06868383, -0.04275973, 0.05169088, -0.02939899, 0.00480521, -0.01130181, -0.00164181, -0.03087403,
            0.07130666, -0.03580481, -0.05381081, 0.00215851, 0.04493982, -0.03930632, 0.02722579, 0.05980525,
            -0.00353011, 0.02689346, -0.00536684, -0.05491739, -0.02888788, -0.05170505, 0.05243583, 0.00778922,
            0.04041063, 0.02485109, 0.03951903, 0.07313389, 0.02834418, -0.01903222, 0.00179730, -0.02860065,
            0.00643315, -0.01877479, 0.01928278, 0.04729762, -0.01982837, -0.01708077, -0.03030094, -0.00872895,
            0.00667370, -0.05643269, 0.00065257, -0.02788488, -0.03016119, 0.00779424, -0.05289216, -0.00277851,
            -0.00858189, 0.00227508, 0.01468778, -0.01659023, 0.02164984, -0.04567245, 0.00472513, 0.00251006,
            -0.00272570, -0.02665212, -0.04194590, -0.05940509, 0.01551652, 0.01752679, 0.04340313, 0.00880810,
            0.00164480, 0.01407121, -0.00617652, 0.00074064, -0.03765314, 0.01225436, 0.00076418, 0.01153753,
            0.05298226, -0.02073503, -0.01609300, -0.04571503, -0.01847511, 0.00276779, 0.01287347, 0.03892899,
            0.05548926, -0.00015479, 0.10774200, 0.01225894, -0.04222156, -0.02496758, 0.02690129, -0.02723961,
            0.00888010, 0.01053329, 0.01103500, -0.05252273, -0.02876102, 0.00545203, 0.03137938, 0.05324582,
            0.02351000, -0.02165280, -0.04151570, 0.04404522, 0.00270667, -0.04216470, 0.03400638, 0.05286333,
            -0.02782471, -0.00692831, -0.01087529, 0.02397397, 0.01908491, 0.10165402, 0.01192315, 0.03323966,
            0.00156531, 0.00684090, -0.04303455, -0.02269479, -0.00703676, 0.04809753, -0.04040154, -0.04474712,
            -0.00435106, -0.00536850, -0.03433465, 0.02644734, 0.03914220, -0.07473983, 0.01391984, 0.04915992,
            -0.01467092, 0.02913694, 0.01605081, -0.02426930, 0.01439036, -0.00334398, -0.04815178, 0.01305333,
            0.05471212, -0.00646846, 0.00527072, 0.03646788, -0.02816276, -0.02160573, -0.01714897, 0.00670109,
            -0.03217531, 0.03011084, -0.05738145, 0.07287889, -0.01226333, -0.01542985, -0.04047069, 0.06292827,
            0.01813532, -0.02326977, 0.02284999, -0.02211571, 0.02848732, 0.01380665, -0.05310283, -0.02185958,
            0.02211477, -0.00869291, -0.03692819, -0.02908471, 0.03007641, 0.01058592, 0.00772493, -0.01741375,
            -0.00406730, 0.03831686, 0.05354239, -0.01951066, 0.08131564, 0.01903587, 0.05849239, 0.03144202,
            -0.02729571, -0.03951886, 0.01532794, 0.01927051, 0.00626895, 0.02179069, -0.00102260, 0.03974926,
            -0.01499388, 0.05027582, 0.00153445, 0.00267263, -0.00862387, 0.03730031, -0.02336266, 0.00211120,
            -0.03577235, 0.04010805, 0.03624576, -0.09867355, 0.00319999, -0.01319205, -0.04574693, -0.02937431,
            -0.07047797, -0.04953273, 0.01261019, -0.04338681, -0.01883920, 0.01700835, 0.00771789, -0.06742840,
            -0.03863352, -0.02221704, -0.03842711, -0.03582061, -0.03585142, 0.04334721, -0.05871584, 0.00967069,
            -0.00793832, -0.01053439, -0.00422087, 0.02071142, -0.03889384, -0.02637660, 0.04231188, 0.04124434,
            0.04199277, 0.05036108, -0.03972300, -0.04513317, -0.01539438, 0.02793515, 0.06088092, 0.06925251,
            -0.05282828, 0.03576118, -0.00228063, 0.01308680, -0.00609524, 0.04946741, 0.06882796, -0.00987220,
            0.00356537, 0.01931496, -0.03486556, -0.06363414, -0.02902352, 0.01857778, -0.02927927, -0.08302981,
            0.03583512, 0.00282543, 0.00702747, -0.00476980, -0.00118030, -0.00239551, 0.04493970, 0.01321466,
            -0.02098357, -0.03754058, -0.04208729, 0.02222012, -0.00191648, 0.04204706, 0.03579066, 0.01945201,
            -0.03115783, 0.01117601, -0.01478992, 0.05169727, -0.00606923, 0.05103835, 0.00597416, 0.06177086,
            0.03747640, -0.04470407, 0.00900277, 0.00593546, -0.08493836, -0.02100805, 0.06306777, -0.03971301,
            -0.04180663, -0.01795717, -0.01849928, -0.05761963, 0.03137821, -0.03301819, 0.00358090, -0.00503534,
            -0.03042374, -0.03990313, -0.02443100, 0.01073140, 0.01850497, 0.02642190, -0.00535564, -0.01626060,
            0.05588126, -0.02401949, 0.01802678, 0.00460109, -0.04322317, -0.01042459, -0.01037219, 0.03160185,
            0.00956813, 0.01986910, -0.00341293, -0.00246329, 0.04939777, -0.01292954, 0.05021381, -0.04974762,
            -0.02004814, 0.03197408, 0.05823653, -0.10174867, -0.02986078, -0.02796665, -0.04566650, -0.00647302,
            0.02874636, -0.02680333, 0.00494275, 0.02021375, 0.01945844, 0.03258825, -0.03995639, 0.02900195,
            -0.02796645, 0.01511639, -0.01478173, 0.02078000, -0.01768648, -0.02590950, -0.02487441, -0.00840603,
            0.05139028, -0.07400550, 0.05012517, -0.03799829, 0.03976471, 0.05454439, 0.01454476, -0.01531853,
            0.02700065, 0.02433223, -0.02162222, -0.00781319, 0.00936916, -0.00627922, 0.04025667, 0.03600691,
            0.03918017, 0.06261952, -0.01092203, -0.03351108, 0.00442382, 0.00721322, -0.00660078, -0.00267135,
            -0.01364039, -0.02826845, -0.02713794, 0.03803687, 0.02149630, 0.00578077, 0.03469257, 0.04230221,
            -0.00993030, -0.00825414, -0.04851964, 0.02312060, 0.02865960, 0.00119022, 0.01859855, 0.00081111,
            0.08854548, -0.06210192, 0.00907954, -0.02899407, -0.01976701, -0.05404822, 0.00094303, 0.01109314,
            0.01103812, -0.00841275, -0.01044191, -0.07068745, -0.02154285, 0.01556709, 0.00627414, -0.01626849,
            -0.01840844, 0.03008935, 0.00429419, -0.00903631, -0.03292655, -0.07754056, -0.02733381, -0.00539867
        ];

        this.initializeEventListeners();
        this.loadModels();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');

        uploadArea.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFile(e.target.files[0]);
            }
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    async loadModels() {
        try {
            this.updateStatus('Loading AI models...', 'loading');
            
            this.faceDetection = new FaceDetection({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
            });
            
            this.faceDetection.setOptions({
                model: 'short',
                minDetectionConfidence: 0.5,
            });
            
            await this.faceDetection.initialize();
            this.isFaceDetectionLoaded = true;
            
            this.session = await ort.InferenceSession.create('./marcus_face_model.onnx');
            this.isModelLoaded = true;
            
            this.updateStatus('✅ AI models ready! Upload an image to start.', 'ready');
        } catch (error) {
            console.error('Failed to load models:', error);
            this.updateStatus('❌ Failed to load AI models. Please refresh the page.', 'error');
        }
    }

    updateStatus(message, type) {
        const statusElement = document.getElementById('status');
        statusElement.innerHTML = message;
        statusElement.className = `status ${type}`;
    }

    validateFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (PNG, JPG, JPEG)');
            return false;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return false;
        }
        return true;
    }

    handleFile(file) {
        if (!this.validateFile(file)) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.hideAllSections();
            this.processImage(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    hideAllSections() {
        document.getElementById('previewSection').style.display = 'none';
        document.getElementById('loading').style.display = 'none';
        document.getElementById('detectionCanvas').style.display = 'none';
        document.getElementById('resultCard').style.display = 'none';
    }

    async processImage(imageSrc) {
        if (!this.isFaceDetectionLoaded || !this.isModelLoaded) {
            alert('AI models are still loading. Please wait.');
            return;
        }

        try {
            document.getElementById('loading').style.display = 'block';
            this.updateLoadingText('Detecting faces...');

            const img = new Image();
            await new Promise((resolve) => {
                img.onload = resolve;
                img.src = imageSrc;
            });

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const detections = await this.detectFaces(canvas);
            
            if (detections.length === 0) {
                this.showResult('👤', 'No faces detected', 'Please try a different image with visible faces', 'error');
                return;
            }

            this.updateLoadingText('Analyzing faces...');
            const results = await this.analyzeAllFaces(img, detections);
            
            if (detections.length === 1) {
                // Single face - show detection canvas and result
                this.showSingleFaceResult(img, detections, results[0]);
            } else {
                // Multiple faces - show detection canvas then results
                this.showMultiFaceResults(img, detections, results);
            }

        } catch (error) {
            console.error('Processing failed:', error);
            this.showResult('⚠️', 'Analysis Failed', 'Please try again with a different image', 'error');
        } finally {
            document.getElementById('loading').style.display = 'none';
        }
    }

    updateLoadingText(text) {
        const loadingText = document.getElementById('loadingText');
        if (loadingText) loadingText.textContent = text;
    }

    async detectFaces(canvas) {
        return new Promise((resolve) => {
            const detections = [];
            this.faceDetection.onResults((results) => {
                if (results.detections) {
                    detections.push(...results.detections);
                }
                resolve(detections);
            });
            this.faceDetection.send({ image: canvas });
        });
    }

    async analyzeAllFaces(img, detections) {
        const results = [];
        for (let i = 0; i < detections.length; i++) {
            const detection = detections[i];
            const faceCanvas = this.extractFace(img, detection);
            const marcusResult = await this.recognizeMarcus(faceCanvas);
            results.push({ faceIndex: i + 1, ...marcusResult });
        }
        return results;
    }

    extractFace(img, detection) {
        const bbox = detection.boundingBox;
        
        // Reduce padding and be more precise with face extraction
        const padding = 0.1; // Reduced from 0.2 to 0.1 (10% instead of 20%)
        
        // Calculate precise face coordinates
        const centerX = bbox.xCenter * img.width;
        const centerY = bbox.yCenter * img.height;
        const faceWidth = bbox.width * img.width;
        const faceHeight = bbox.height * img.height;
        
        // Add padding but keep it more conservative
        const paddedWidth = faceWidth * (1 + padding);
        const paddedHeight = faceHeight * (1 + padding);
        
        // Calculate extraction coordinates
        const x = Math.max(0, centerX - paddedWidth / 2);
        const y = Math.max(0, centerY - paddedHeight / 2);
        const width = Math.min(img.width - x, paddedWidth);
        const height = Math.min(img.height - y, paddedHeight);

        // Create face canvas with better aspect ratio handling
        const faceCanvas = document.createElement('canvas');
        faceCanvas.width = 160;
        faceCanvas.height = 160;
        const ctx = faceCanvas.getContext('2d');
        
        // Fill with neutral background first
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, 160, 160);
        
        // Calculate scaling to maintain aspect ratio
        const scale = Math.min(160 / width, 160 / height);
        const scaledWidth = width * scale;
        const scaledHeight = height * scale;
        const offsetX = (160 - scaledWidth) / 2;
        const offsetY = (160 - scaledHeight) / 2;
        
        // Draw the face with proper scaling
        ctx.drawImage(img, x, y, width, height, offsetX, offsetY, scaledWidth, scaledHeight);
        
        // Debug: log extraction details
        console.log('Face extraction:', {
            originalBbox: bbox,
            extractedRegion: { x, y, width, height },
            centerPoint: { centerX, centerY },
            faceDimensions: { faceWidth, faceHeight }
        });
        
        return faceCanvas;
    }

    async recognizeMarcus(faceCanvas) {
        try {
            const ctx = faceCanvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, 160, 160);
            const data = imageData.data;
            
            const input = new Float32Array(3 * 160 * 160);
            for (let i = 0; i < 160 * 160; i++) {
                const pixelIndex = i * 4;
                const r = data[pixelIndex] / 255.0;
                const g = data[pixelIndex + 1] / 255.0;
                const b = data[pixelIndex + 2] / 255.0;
                
                input[i] = (r - 0.5) / 0.5;
                input[160 * 160 + i] = (g - 0.5) / 0.5;
                input[160 * 160 * 2 + i] = (b - 0.5) / 0.5;
            }
            
            const inputTensor = new ort.Tensor('float32', input, [1, 3, 160, 160]);
            const results = await this.session.run({ input: inputTensor });
            const output = results.output;
            const embedding = Array.from(output.data);

            const similarity = this.cosineSimilarity(embedding, this.marcusMeanEmbedding);
            const confidence = Math.max(0, (similarity - this.threshold) / (1 - this.threshold) * 100);
            const isMarcus = similarity >= this.threshold && confidence >= this.minConfidence;

            console.log('Face Analysis:', { similarity, confidence, isMarcus });
            return { similarity, confidence, isMarcus };

        } catch (error) {
            console.error('Marcus recognition failed:', error);
            return { similarity: 0, confidence: 0, isMarcus: false };
        }
    }

    showSingleFaceResult(img, detections, result) {
        // Show detection canvas
        this.drawDetectionCanvas(img, detections, [result]);
        document.getElementById('detectionCanvas').style.display = 'block';
        
        setTimeout(() => {
            if (result.isMarcus) {
                this.showResult('🎉', 'MARCUS DETECTED!', 'This is definitely Marcus', 'marcus');
            } else {
                this.showResult('🔍', 'Not Marcus', 'This person is not Marcus', 'not-marcus');
            }
        }, 500);
    }

    showMultiFaceResults(img, detections, results) {
        // Show detection canvas
        this.drawDetectionCanvas(img, detections, results);
        document.getElementById('detectionCanvas').style.display = 'block';
        
        const marcusResults = results.filter(r => r.isMarcus);
        
        setTimeout(() => {
            if (marcusResults.length > 0) {
                const marcusFaceIds = marcusResults.map(r => r.faceIndex).join(', ');
                const subtitle = marcusResults.length === 1 
                    ? `Found Marcus in face ${marcusFaceIds}`
                    : `Found Marcus in faces ${marcusFaceIds}`;
                this.showResult('🎉', 'MARCUS DETECTED!', subtitle, 'marcus');
            } else {
                this.showResult('🔍', 'Marcus not found', 
                    `Analyzed ${results.length} faces in this image`, 'not-marcus');
            }
        }, 500);
    }

    drawDetectionCanvas(img, detections, results = []) {
        const canvas = document.getElementById('faceDetectionCanvas');
        const canvasSize = 350;
        canvas.width = canvasSize;
        canvas.height = canvasSize;
        const ctx = canvas.getContext('2d');

        const scale = Math.min(canvasSize / img.width, canvasSize / img.height);
        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;
        const offsetX = (canvasSize - scaledWidth) / 2;
        const offsetY = (canvasSize - scaledHeight) / 2;

        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, canvasSize, canvasSize);
        ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

        ctx.lineWidth = 4;
        ctx.font = 'bold 16px Arial';
        ctx.shadowBlur = 3;

        detections.forEach((detection, index) => {
            const bbox = detection.boundingBox;
            const x = (bbox.xCenter * img.width - (bbox.width * img.width) / 2) * scale + offsetX;
            const y = (bbox.yCenter * img.height - (bbox.height * img.height) / 2) * scale + offsetY;
            const width = bbox.width * img.width * scale;
            const height = bbox.height * img.height * scale;

            // Check if this face is Marcus
            const result = results[index];
            const isMarcus = result && result.isMarcus;
            
            if (isMarcus) {
                // Marcus face - green
                ctx.strokeStyle = '#10b981';
                ctx.fillStyle = '#10b981';
                ctx.shadowColor = 'rgba(16, 185, 129, 0.3)';
            } else {
                // Not Marcus - blue
                ctx.strokeStyle = '#3b82f6';
                ctx.fillStyle = '#3b82f6';
                ctx.shadowColor = 'rgba(59, 130, 246, 0.3)';
            }

            ctx.strokeRect(x, y, width, height);
            const label = isMarcus ? `Face ${index + 1} (Marcus!)` : `Face ${index + 1}`;
            ctx.fillText(label, x, y - 8);
        });
        
        document.getElementById('detectionInfo').textContent = 
            `Detected ${detections.length} face${detections.length > 1 ? 's' : ''}`;
    }

    showResult(icon, title, subtitle, type) {
        const resultCard = document.getElementById('resultCard');
        const resultIcon = document.getElementById('resultIcon');
        const resultText = document.getElementById('resultText');
        const resultSubtext = document.getElementById('resultSubtext');

        resultIcon.textContent = icon;
        resultText.textContent = title;
        resultSubtext.textContent = subtitle;
        
        resultCard.className = `result-card ${type}`;
        resultCard.style.display = 'block';
    }

    cosineSimilarity(a, b) {
        let dotProduct = 0, magnitudeA = 0, magnitudeB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }
        magnitudeA = Math.sqrt(magnitudeA);
        magnitudeB = Math.sqrt(magnitudeB);
        return magnitudeA === 0 || magnitudeB === 0 ? 0 : dotProduct / (magnitudeA * magnitudeB);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.marcusApp = new MarcusFaceRecognition();
});

console.log('🚀 Marcus Face Recognition App with MediaPipe Face Detection');
console.log('✅ Cosine similarity threshold: 0.2292 | Min confidence: 36%');
console.log('🤖 MediaPipe will detect faces automatically');
console.log('📊 Check console for detailed analysis metrics');
