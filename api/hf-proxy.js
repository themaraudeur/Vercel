const MODEL = 'timbrooks/instruct-pix2pix';
const HF_URL = `https://api-inference.huggingface.co/models/${MODEL}`;

export default async function handler(req, res) {
  // Gestion des en-têtes CORS (comme sur ton ancien setup)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).send('Méthode non autorisée');
  }

  try {
    const { image, prompt, token } = req.body;

    if (!image || !prompt || !token) {
      return res.status(400).send('Données manquantes (image, prompt ou token)');
    }

    // Nettoyage standard du Base64
    const base64Data = image.includes(',') ? image.split(',')[1] : image;

    // Appel direct à Hugging Face
    const hfResp = await fetch(HF_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        inputs: base64Data,
        parameters: {
          prompt: prompt,
          num_inference_steps: 20,
          image_guidance_scale: 1.5
        }
      })
    });

    if (!hfResp.ok) {
      const errText = await hfResp.text();
      return res.status(hfResp.status).send(errText || `Erreur HF: ${hfResp.status}`);
    }

    // Récupération de l'image renvoyée par le modèle
    const arrayBuffer = await hfResp.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Renvoi de l'image traitée avec le bon Content-Type
    const contentType = hfResp.headers.get('content-type') || 'image/jpeg';
    res.setHeader('Content-Type', contentType);
    return res.status(200).send(buffer);

  } catch (err) {
    return res.status(500).send('Erreur Proxy: ' + err.message);
  }
}
