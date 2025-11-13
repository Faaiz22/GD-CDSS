
import requests
from Bio import SeqIO
from io import StringIO

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
HEADERS = {"User-Agent": "GeneDrugCDSSv2/1.0 (Contact: example@example.com)"}


def get_fasta_from_gene_symbol(symbol: str, email: str = "example@example.com") -> str:
    """
    Fetch protein FASTA for a gene symbol using NCBI E-Utilities.
    Workflow:
      1. esearch(db=gene) → UID
      2. esummary → Extract preferred RefSeq protein accession
      3. efetch(db=protein) → FASTA
    Returns raw FASTA text.
    Raises exceptions on failure.
    """

    # Step 1: Search NCBI Gene
    search_url = (
        f"{NCBI_EUTILS}esearch.fcgi?db=gene&term={symbol}[sym]&retmode=json&email={email}"
    )
    resp = requests.get(search_url, headers=HEADERS)
    if resp.status_code != 200:
        raise Exception(f"NCBI esearch failed for symbol {symbol}")

    data = resp.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise Exception(f"No NCBI Gene record found for symbol {symbol}")

    gene_uid = ids[0]

    # Step 2: esummary to find RefSeq protein accession
    summary_url = (
        f"{NCBI_EUTILS}esummary.fcgi?db=gene&id={gene_uid}&retmode=json&email={email}"
    )
    resp = requests.get(summary_url, headers=HEADERS)
    if resp.status_code != 200:
        raise Exception(f"NCBI esummary failed for UID {gene_uid}")

    summaries = resp.json()
    doc = summaries.get("result", {}).get(gene_uid, {})
    accessions = []

    # Extract protein accessions from genomic info blocks
    for gi in doc.get("genomicinfo", []):
        acc = gi.get("protacc")
        if acc:
            accessions.append(acc)

    if not accessions:
        raise Exception(f"No protein accession found for symbol {symbol}")

    prot_acc = accessions[0]

    # Step 3: efetch → FASTA
    fetch_url = (
        f"{NCBI_EUTILS}efetch.fcgi?db=protein&id={prot_acc}&rettype=fasta&retmode=text&email={email}"
    )
    resp = requests.get(fetch_url, headers=HEADERS)
    if resp.status_code != 200:
        raise Exception(f"NCBI efetch failed for protein accession {prot_acc}")

    fasta_text = resp.text.strip()
    if not fasta_text.startswith(">"):
        raise Exception(f"Invalid FASTA received for {symbol}")

    return fasta_text


def parse_fasta_to_sequence(fasta_text: str) -> str:
    """
    Convert FASTA string → amino acid sequence string.
    """
    handle = StringIO(fasta_text)
    for record in SeqIO.parse(handle, "fasta"):
        return str(record.seq)
    raise Exception("Unable to parse FASTA text into a sequence")

