#!/usr/bin/env python3
"""
Script d'extraction de texte depuis un PDF pour le pipeline RAG.
Extrait le texte de chaque page et le sauvegarde dans un fichier .txt formaté.
"""

import sys
import argparse
from pathlib import Path
import fitz  # PyMuPDF


def extract_pdf(pdf_path, txt_path):
    """
    Extrait le texte d'un PDF et le sauvegarde dans un fichier texte.
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF
        txt_path (str): Chemin vers le fichier de sortie .txt
    
    Returns:
        tuple: (nombre_pages_traitees, nombre_pages_ignorees)
    """
    pdf_path = Path(pdf_path)
    txt_path = Path(txt_path)
    
    # Vérifier que le fichier PDF existe
    if not pdf_path.exists():
        raise FileNotFoundError(f"Le fichier PDF '{pdf_path}' n'existe pas.")
    
    # Ouvrir le PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise Exception(f"Erreur lors de l'ouverture du PDF: {e}")
    
    pages_processed = 0
    pages_ignored = 0
    extracted_text = []
    
    print(f"📄 Traitement du PDF: {pdf_path.name}")
    print(f"📊 Nombre total de pages: {len(doc)}")
    
    # Extraire le texte de chaque page
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        
        # Compter les mots (approximation simple)
        word_count = len(text.split()) if text else 0
        
        # Ignorer les pages vides ou quasi-vides (< 20 mots)
        if word_count < 20:
            print(f"⚠️  Page {page_num + 1} ignorée (seulement {word_count} mots)")
            pages_ignored += 1
            continue
        
        # Ajouter le texte formaté
        extracted_text.append(f"<PAGE {page_num + 1}>\n{text}\n")
        pages_processed += 1
        print(f"✅ Page {page_num + 1} traitée ({word_count} mots)")
    
    # Fermer le document
    doc.close()
    
    # Sauvegarder le texte extrait
    if extracted_text:
        # Créer le répertoire de sortie si nécessaire
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Écrire le fichier de sortie en UTF-8
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(extracted_text))
        
        print(f"\n💾 Texte sauvegardé dans: {txt_path}")
        print(f"📈 Pages traitées: {pages_processed}")
        print(f"🚫 Pages ignorées: {pages_ignored}")
    else:
        print("⚠️  Aucun texte valide trouvé dans le PDF.")
    
    return pages_processed, pages_ignored


def main():
    """Interface en ligne de commande pour le script."""
    parser = argparse.ArgumentParser(
        description="Extrait le texte d'un PDF et le sauvegarde dans un fichier .txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python src/pdf_to_text.py cours.pdf sortie.txt
  python src/pdf_to_text.py data/mon_document.pdf data/extrait.txt
        """
    )
    
    parser.add_argument(
        'pdf_path',
        help='Chemin vers le fichier PDF à traiter'
    )
    
    parser.add_argument(
        'txt_path',
        help='Chemin vers le fichier de sortie .txt'
    )
    
    args = parser.parse_args()
    
    try:
        pages_processed, pages_ignored = extract_pdf(args.pdf_path, args.txt_path)
        
        if pages_processed > 0:
            print(f"\n🎉 Extraction terminée avec succès!")
            print(f"📄 Fichier de sortie: {args.txt_path}")
            print(f"📊 {pages_processed} pages traitées, {pages_ignored} pages ignorées")
        else:
            print("\n❌ Aucune page n'a pu être traitée.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
