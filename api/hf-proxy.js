const MODEL = 'timbrooks/instruct-pix2pix';
const HF_URL = `https://api-inference.huggingface.co/models/${MODEL}`;

export default async function handler(req, res) {
  // Gestion des requêtes CORS / Preflight
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).send('Method not allowed');
  }

  const { image, prompt, token } = req.body;
  if (!image || !prompt || !token) {
    return res.status(400).send('Missing image, prompt, or token');
  }

  // Nettoyage du préfixe Base64 si existant
  const base64 = image.includes(',') ? image.split(',')[1] : image;

  try {
    const hfResp = await fetch(HF_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        inputs: base64,
        parameters: {
          prompt: prompt,
          num_inference_steps: 20,
          image_guidance_scale: 1.5
        }
      })
    });

    if (!hfResp.ok) {
      const errText = await hfResp.text();
      return res.status(hfResp.status).send(errText || `Hugging Face a répondu ${hfResp.status}`);
    }

    const contentType = hfResp.headers.get('content-type') || 'image/jpeg';
    const arrayBuffer = await hfResp.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    res.setHeader('Content-Type', contentType);
    return res.status(200).send(buffer);

  } catch (err) {
    return res.status(500).send('Erreur proxy : ' + err.message);
  }
}
